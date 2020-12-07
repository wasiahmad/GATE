"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from prettytable import PrettyTable

from clie.modules import Embedder
from clie.objects.tree import inputs_to_tree_reps, tree_to_dist_mat
from clie.inputters import constant
from clie.encoders import TransformerEncoder


class GTNClassifier(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.model = GTNRelationModel(opt)
        self.classifier = nn.Linear(opt.out_dim, opt.label_size)

    def forward(self, **kwargs):
        outputs, pooled_out = self.model(**kwargs)
        logits = self.classifier(outputs)
        return logits, pooled_out

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

    def forward(self, input, input_len, adj_mask=None, struct_relpos=None):
        layer_outputs, _ = self.transformer(input,
                                            adj_mask=adj_mask,
                                            lengths=input_len,
                                            struct_relpos=struct_relpos)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class GTNRelationModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.pool_type = opt.pool_type
        self.prune_k = opt.prune_k
        self.embedding = Embedder(opt)

        self.transform = None
        if opt.emsize != opt.tran_hid:
            # to match with Transformer Encoder size, we apply transform
            self.transform = nn.Linear(opt.emsize, opt.tran_hid)

        self.embed_graph = opt.embed_graph
        self.max_tree_dist = opt.max_tree_dist
        self.num_head = opt.num_head
        if self.embed_graph > 0:
            assert len(self.max_tree_dist) == self.embed_graph
            assert self.embed_graph <= self.num_head

        self.encoder = Encoder(opt)
        opt.out_dim = opt.tran_hid

        # gcn layer
        self.use_gcn = opt.gcn_hid > 0
        if self.use_gcn:
            self.W = nn.ModuleList()
            for layer in range(opt.gcn_layers):
                input_dim = opt.out_dim if layer == 0 else opt.gcn_hid
                self.W.append(nn.Linear(input_dim, opt.gcn_hid))
            opt.out_dim = opt.gcn_hid
            self.gcn_drop = nn.Dropout(opt.dropout_gcn)

        # self.use_sent_rep = opt.use_sent_rep
        self.use_sent_rep = opt.use_sent_rep if hasattr(opt, 'use_sent_rep') else True
        in_dim = opt.out_dim * 3 if self.use_sent_rep else opt.out_dim * 2
        # output mlp layers
        layers = [nn.Linear(in_dim, opt.out_dim), nn.ReLU()]
        for _ in range(opt.mlp_layers - 1):
            layers += [nn.Linear(opt.out_dim, opt.out_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, **kwargs):
        sent_len = kwargs.get('len_rep')
        words = kwargs.get('word_rep')
        head = kwargs.get('head_rep')
        subj_pos = kwargs.get('subj_rep')
        obj_pos = kwargs.get('obj_rep')

        embs = self.embedding(**kwargs)
        if self.transform is not None:
            embs = self.transform(embs)

        adj_mask = kwargs.get('adj_mask')
        if self.embed_graph > 0:
            if adj_mask is None:
                # adj_mask = B, max_len, max_len
                adj_mask = inputs_to_tree_reps(head, words, sent_len, self.prune_k,
                                               subj_pos, obj_pos, self_loop=True,
                                               fn=tree_to_dist_mat)
                adj_mask = torch.from_numpy(adj_mask)

            adj_mask_list = []
            for k in range(self.embed_graph):
                mask_k = torch.empty_like(adj_mask).copy_(adj_mask)
                mask_k[mask_k > self.max_tree_dist[k]] = 0
                adj_mask_list.append(mask_k.to(embs))

            no_mask_count = self.num_head - self.embed_graph
            if no_mask_count > 0:
                no_mask = ~words.eq(constant.PAD)
                no_mask = no_mask.unsqueeze(1).repeat(1, words.size(1), 1).float()
                adj_mask_list += [no_mask] * no_mask_count

            assert len(adj_mask_list) == self.num_head
            # B, num_head, max_len, max_len
            adj_mask = torch.stack(adj_mask_list, axis=1)

        struct_relpos = kwargs.get('struct_relpos')
        memory_bank, layer_outputs = self.encoder(embs, sent_len, adj_mask,
                                                  struct_relpos=struct_relpos)

        # gcn layer
        if self.use_gcn:
            adj = inputs_to_tree_reps(head, words, sent_len,
                                      self.prune_k, subj_pos, obj_pos)
            # batch_size x max_len x max_len
            adj = torch.from_numpy(adj).to(embs)
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

        # pooling
        masks = words.eq(constant.PAD)
        subj_mask, obj_mask = ~subj_pos.eq(0), ~obj_pos.eq(0)  # invert mask
        subj_mask = (subj_mask | masks).unsqueeze(2)  # logical or with word masks
        obj_mask = (obj_mask | masks).unsqueeze(2)

        h_out = pool(memory_bank, masks.unsqueeze(2), type=self.pool_type)
        subj_out = pool(memory_bank, subj_mask, type=self.pool_type)
        obj_out = pool(memory_bank, obj_mask, type=self.pool_type)
        if self.use_sent_rep:
            outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        else:
            outputs = torch.cat([subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out


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
