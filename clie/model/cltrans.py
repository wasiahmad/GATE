import torch
import torch.nn as nn
import torch.nn.functional as f

from clie.modules import Embedder
from clie.objects.tree import inputs_to_tree_reps
from clie.inputters import constant
from clie.encoders import TransformerEncoder
from prettytable import PrettyTable


class ClTransClassifier(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.model = InfoExtractionModel(opt)
        self.classifier = nn.Linear(opt.out_dim, opt.label_size)

    def forward(self, **kwargs):
        main_out, knn_out = self.model(**kwargs)
        main_logits = self.classifier(main_out)
        knn_logits = None
        if knn_out is not None:
            knn_logits = self.classifier(knn_out)
        return main_logits, knn_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #", "Train?"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        table.align["Trainable"] = "c"
        for name, parameters in self.named_parameters():
            is_trainable = 'Yes' if parameters.requires_grad else 'No'
            table.add_row([name, str(list(parameters.shape)), parameters.numel(), is_trainable])
        return table


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        if len(args.max_relative_pos) != args.tran_layers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.tran_layers

        self.transformer = TransformerEncoder(num_layers=args.tran_layers,
                                              d_model=args.tran_hid,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(args.tran_hid, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self, input, input_len, adj_mask=None):
        layer_outputs, _ = self.transformer(input,
                                            adj_mask=adj_mask,
                                            lengths=input_len)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class InfoExtractionModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.pool_type = opt.pool_type
        self.prune_k = opt.prune_k
        self.embedding = Embedder(opt)

        assert opt.position_dim == 0  # we only use relative position representation

        self.f_transform = None
        self.b_transform = None
        if opt.emsize != opt.tran_hid:
            # to match with Transformer Encoder size, we apply transform
            self.f_transform = nn.Linear(opt.emsize, opt.tran_hid)
            self.b_transform = nn.Linear(opt.tran_hid, opt.emsize)

        self.encoder = Encoder(opt)
        opt.out_dim = opt.emsize

        self.selection = nn.Linear(opt.emsize * 2, 1)

        # gcn layer
        self.use_gcn = opt.gcn_hid > 0
        if self.use_gcn:
            self.W = nn.ModuleList()
            for layer in range(opt.gcn_layers):
                input_dim = opt.out_dim if layer == 0 else opt.gcn_hid
                self.W.append(nn.Linear(input_dim, opt.gcn_hid))
            opt.out_dim = opt.gcn_hid
            self.gcn_drop = nn.Dropout(opt.dropout_gcn)

        # output mlp layers
        self.use_sent_rep = opt.use_sent_rep
        in_dim = opt.out_dim * 3 if self.use_sent_rep else opt.out_dim * 2
        layers = [nn.Linear(in_dim, opt.out_dim), nn.ReLU()]
        for _ in range(opt.mlp_layers - 1):
            layers += [nn.Linear(opt.out_dim, opt.out_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def selective_attention(self, knn_words, main_embeds, sent_len, **kwargs):
        batch_size, max_len, knn_size = knn_words.size()
        knn_val = knn_words.size(2)
        pos_rep = kwargs.get('pos_rep')
        ner_rep = kwargs.get('ner_rep')
        deprel_rep = kwargs.get('deprel_rep')
        type_rep = kwargs.get('type_rep')
        if pos_rep is not None:
            pos_rep = pos_rep.unsqueeze(-1).repeat(1, 1, knn_size)
            pos_rep = pos_rep.reshape(batch_size * max_len, -1)
        if ner_rep is not None:
            ner_rep = ner_rep.unsqueeze(-1).repeat(1, 1, knn_size)
            ner_rep = ner_rep.reshape(batch_size * max_len, -1)
        if deprel_rep is not None:
            deprel_rep = deprel_rep.unsqueeze(-1).repeat(1, 1, knn_size)
            deprel_rep = deprel_rep.reshape(batch_size * max_len, -1)
        if type_rep is not None:
            type_rep = type_rep.unsqueeze(-1).repeat(1, 1, knn_size)
            type_rep = type_rep.reshape(batch_size * max_len, -1)

        # knn_embs: bsz*max_len x knn_size x emb_dim
        knn_embs = self.embedding(
            word_rep=knn_words.reshape(batch_size * max_len, -1),
            pos_rep=pos_rep, ner_rep=ner_rep, deprel_rep=deprel_rep, type_rep=type_rep
        )
        if self.f_transform is not None:
            main_embeds = self.f_transform(main_embeds)
        contextual_embeds, _ = self.encoder(main_embeds, sent_len)
        if self.b_transform is not None:
            contextual_embeds = self.b_transform(contextual_embeds)

        ##### Perform selective attention
        # check Eq. (6) and (7) from paper - https://www.aclweb.org/anthology/D19-1068.pdf
        contextual_embeds = contextual_embeds.unsqueeze(2).repeat(1, 1, knn_val, 1)
        attention = self.selection(torch.cat([
            contextual_embeds,
            knn_embs.reshape(batch_size, max_len, knn_val, -1)
        ], dim=-1))
        attention = attention.squeeze(3)  # bsz x max_len x knn_val
        attention = f.softmax(attention, dim=-1)
        selected_index = attention.max(dim=-1)[1]  # bsz x max_len
        knn_embs = batched_index_select(knn_embs, 1, selected_index)
        memory_bank = knn_embs.reshape(batch_size, max_len, -1)
        return memory_bank

    def apply_gcn(self, adj, memory_bank):
        # batch_size x max_len x max_len
        adj = torch.from_numpy(adj).to(memory_bank)
        denom = adj.sum(2).unsqueeze(2) + 1
        num_layers = len(self.W)
        for l in range(num_layers):
            # adj = b x max_len x max_len
            Ax = adj.bmm(memory_bank)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](memory_bank)  # self loop
            AxW = AxW / denom
            gAxW = f.relu(AxW)
            memory_bank = self.gcn_drop(gAxW) if l < num_layers - 1 else gAxW
        return memory_bank

    def compute_outputs(self, memory_bank, masks, subj_mask, obj_mask):
        h_out = pool(memory_bank, masks.unsqueeze(2), type=self.pool_type)
        subj_out = pool(memory_bank, subj_mask, type=self.pool_type)
        obj_out = pool(memory_bank, obj_mask, type=self.pool_type)
        if self.use_sent_rep:
            outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        else:
            outputs = torch.cat([subj_out, obj_out], dim=1)
        return self.out_mlp(outputs)

    def forward(self, **kwargs):
        sent_len = kwargs.get('len_rep')
        words = kwargs.get('word_rep')
        knn_words = kwargs.get('knn_rep')  # bsz x max_len x knn_size
        head = kwargs.get('head_rep')
        subj_pos = kwargs.get('subj_rep')
        obj_pos = kwargs.get('obj_rep')

        main_embeds = self.embedding(**kwargs)
        knn_embeds = None
        if knn_words is not None:
            knn_embeds = self.selective_attention(
                knn_words, main_embeds, sent_len, **kwargs
            )

        if self.use_gcn:
            adj = inputs_to_tree_reps(head, words, sent_len,
                                      self.prune_k, subj_pos, obj_pos)
            main_embeds = self.apply_gcn(adj, main_embeds)
            if knn_embeds is not None:
                knn_embeds = self.apply_gcn(adj, knn_embeds)

        # prepare masks
        masks = words.eq(constant.PAD)
        subj_mask, obj_mask = ~subj_pos.eq(0), ~obj_pos.eq(0)  # invert mask
        subj_mask = (subj_mask | masks).unsqueeze(2)  # logical or with word masks
        obj_mask = (obj_mask | masks).unsqueeze(2)

        main_out = self.compute_outputs(main_embeds, masks, subj_mask, obj_mask)
        knn_out = None
        if knn_embeds is not None:
            knn_out = self.compute_outputs(knn_embeds, masks, subj_mask, obj_mask)

        return main_out, knn_out


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)
