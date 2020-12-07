"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from clie.modules import Embedder
from clie.objects.tree import inputs_to_tree_reps
from clie.inputters import constant
from clie.encoders import RNNEncoder
from prettytable import PrettyTable


class GCNClassifier(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.model = GCNRelationModel(opt)
        self.classifier = nn.Linear(opt.in_dim, opt.label_size)

    def conv_l2(self):
        return self.model.gcn.conv_l2()

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


class GCNRelationModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.pool_type = opt.pool_type
        self.prune_k = opt.prune_k
        self.embedding = Embedder(opt)

        # gcn layer
        self.no_adj = opt.no_adj
        self.gcn = GCN(opt)

        if opt.gcn_hid > 0:
            opt.in_dim = opt.gcn_hid
        elif opt.rnn_hid > 0:
            opt.in_dim = opt.rnn_hid
        else:
            opt.in_dim = opt.emsize

        # output mlp layers
        self.use_sent_rep = opt.use_sent_rep
        in_dim = opt.in_dim * 3 if self.use_sent_rep else opt.in_dim * 2
        layers = [nn.Linear(in_dim, opt.in_dim), nn.ReLU()]
        for _ in range(opt.mlp_layers - 1):
            layers += [nn.Linear(opt.in_dim, opt.in_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, **kwargs):
        sent_len = kwargs.get('len_rep')
        words = kwargs.get('word_rep')
        head = kwargs.get('head_rep')
        subj_pos = kwargs.get('subj_rep')
        obj_pos = kwargs.get('obj_rep')
        batch_size = words.size(0)

        if self.no_adj:
            maxlen = words.size(1)  # max_sent_len in a batch
            adj = np.zeros((batch_size, maxlen, maxlen))
        else:
            adj = inputs_to_tree_reps(head, words, sent_len,
                                      self.prune_k, subj_pos, obj_pos)

        embs = self.embedding(**kwargs)
        # batch_size x max_len x max_len
        adj = torch.from_numpy(adj).to(embs)
        h = self.gcn(adj, embs, sent_len)

        # pooling
        masks = words.eq(constant.PAD)
        subj_mask, obj_mask = ~subj_pos.eq(0), ~obj_pos.eq(0)  # invert mask
        subj_mask = (subj_mask | masks).unsqueeze(2)  # logical or with word masks
        obj_mask = (obj_mask | masks).unsqueeze(2)

        h_out = pool(h, masks.unsqueeze(2), type=self.pool_type)
        subj_out = pool(h, subj_mask, type=self.pool_type)
        obj_out = pool(h, obj_mask, type=self.pool_type)
        if self.use_sent_rep:
            outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        else:
            outputs = torch.cat([subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, opt):
        super(GCN, self).__init__()
        self.layers = opt.gcn_layers
        self.use_cuda = opt.cuda
        self.no_adj = opt.no_adj
        input_dim = opt.emsize

        # rnn layer
        self.use_rnn = opt.rnn_hid > 0
        if self.use_rnn:
            self.encoder = RNNEncoder(opt.rnn_type,
                                      input_dim,
                                      True,
                                      opt.rnn_layers,
                                      opt.rnn_hid,
                                      opt.dropout_rnn,
                                      use_last=True)
            input_dim = opt.rnn_hid
            self.rnn_drop = nn.Dropout(opt.dropout_rnn)  # use on last layer output

        # gcn layer
        self.use_gcn = opt.gcn_hid > 0
        if self.use_gcn:
            self.gcn_drop = nn.Dropout(opt.dropout_gcn)
            self.W = nn.ModuleList()
            for layer in range(self.layers):
                input_dim = input_dim if layer == 0 else opt.gcn_hid
                self.W.append(nn.Linear(input_dim, opt.gcn_hid))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, input, input_len):
        # rnn_outputs: batch_size x seq_len x nhid*nlayers
        _, rnn_outputs = self.encoder(input, input_len)
        rnn_outputs = self.rnn_drop(rnn_outputs)
        return rnn_outputs

    def forward(self, adj, input, input_len):
        gcn_inputs = input  # b x max_len x nhid
        if self.use_rnn:
            gcn_inputs = self.encode_with_rnn(gcn_inputs, input_len)

        # gcn layer
        if self.use_gcn:
            denom = adj.sum(2).unsqueeze(2) + 1
            for l in range(self.layers):
                # adj = b x max_len x max_len
                Ax = adj.bmm(gcn_inputs)
                AxW = self.W[l](Ax)
                AxW = AxW + self.W[l](gcn_inputs)  # self loop
                AxW = AxW / denom
                gAxW = f.relu(AxW)
                gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs


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
