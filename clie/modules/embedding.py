import torch
import torch.nn as nn
from clie.inputters import constant


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()

        self.word_emb = None
        self.pos_emb = None
        self.ner_emb = None
        self.deprel_emb = None
        self.type_emb = None
        self.position_emb = None

        if args.use_word:
            self.word_emb = nn.Embedding(
                args.word_size, args.word_dim, padding_idx=constant.PAD)
            self.word_emb.weight.data[2:, :].uniform_(
                -constant.EMB_INIT_RANGE, constant.EMB_INIT_RANGE)
        if args.pos_dim > 0:
            self.pos_emb = nn.Embedding(
                args.pos_size, args.pos_dim, padding_idx=constant.PAD)
        if args.ner_dim > 0:
            self.ner_emb = nn.Embedding(
                args.ner_size, args.ner_dim, padding_idx=constant.PAD)
        if args.deprel_dim > 0:
            self.deprel_emb = nn.Embedding(
                args.deprel_size, args.deprel_dim, padding_idx=constant.PAD)
        if args.type_dim > 0:
            self.type_emb = nn.Embedding(
                args.type_size, args.type_dim, padding_idx=constant.PAD)
        if args.position_dim > 0:
            self.position_emb = nn.Embedding(
                args.max_src_len + 1, args.position_dim, padding_idx=args.max_src_len)

        self.drop_in = nn.Dropout(args.dropout_emb)
        emb_dims = [args.bert_dim, args.word_dim, args.pos_dim, args.ner_dim,
                    args.deprel_dim, args.type_dim, args.position_dim]

        self.comb_emb_type = args.comb_emb_type
        if self.comb_emb_type == 'concat':
            args.emsize = sum(emb_dims)
        elif self.comb_emb_type == 'add':
            args.emsize = max(emb_dims)
            nonzero_dims = [i == args.emsize for i in emb_dims if i != 0]
            assert all(nonzero_dims)

    def forward(self, **kwargs):
        words = kwargs.get('word_rep')
        pos = kwargs.get('pos_rep')
        ner = kwargs.get('ner_rep')
        deprel = kwargs.get('deprel_rep')
        type = kwargs.get('type_rep')
        bert_embeds = kwargs.get('bert_embeds')
        positions = kwargs.get('position_rep')

        embs = []
        if bert_embeds is not None:
            embs += [bert_embeds]
        if self.word_emb is not None:
            embs += [self.word_emb(words)]
        if self.pos_emb is not None:
            embs += [self.pos_emb(pos)]
        if self.ner_emb is not None:
            embs += [self.ner_emb(ner)]
        if self.deprel_emb is not None:
            embs += [self.deprel_emb(deprel)]
        if self.type_emb is not None:
            embs += [self.type_emb(type)]
        if self.position_emb is not None:
            if positions is None:
                positions = torch.arange(words.size(1)).to(words)
                positions = positions.expand(*words.size())
            embs += [self.position_emb(positions)]

        if self.comb_emb_type == 'concat':
            embs = torch.cat(embs, dim=-1)
        elif self.comb_emb_type == 'add':
            embs = torch.stack(embs, dim=3).sum(dim=3)

        embs = self.drop_in(embs)
        return embs
