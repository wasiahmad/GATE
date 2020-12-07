# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py
""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from clie.utils.misc import generate_relative_positions_matrix, \
    relative_matmul


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self,
                 head_count,
                 model_dim,
                 d_k,
                 d_v,
                 dropout=0.1,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(MultiHeadedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v

        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)

        self.max_relative_positions = max_relative_positions
        self.use_neg_dist = use_neg_dist

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1 \
                if self.use_neg_dist else max_relative_positions + 1
            self.relative_positions_embeddings_k = nn.Embedding(
                vocab_size, self.d_k)
            self.relative_positions_embeddings_v = nn.Embedding(
                vocab_size, self.d_v)

    def forward(self,
                key,
                value,
                query,
                mask=None,
                **kwargs):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        batch_size = key.size(0)
        head_count = self.head_count
        adj_mask = kwargs.get('adj_mask')
        struct_relpos = kwargs.get('struct_relpos')

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        key = shape(self.key(key), self.d_k)  # bsz x nhead x key_len x d_k
        value = shape(self.value(value), self.d_v)  # bsz x nhead x key_len x d_v
        query = shape(self.query(query), self.d_k)  # bsz x nhead x query_len x d_k

        using_strct_relpos = False
        if self.max_relative_positions > 0:
            if struct_relpos is None:
                key_len = key.size(2)
                # key_len x key_len
                relative_positions_matrix = generate_relative_positions_matrix(
                    key_len, self.max_relative_positions, self.use_neg_dist)
            else:
                using_strct_relpos = True
                # struct_relpos: bsz x key_len x key_len
                dist_matrix = torch.clamp(struct_relpos,
                                          min=-self.max_relative_positions,
                                          max=self.max_relative_positions)
                # Shift values to be >= 0
                if self.use_neg_dist:
                    relative_positions_matrix = dist_matrix + self.max_relative_positions
                else:
                    relative_positions_matrix = torch.abs(dist_matrix)

            #  (bsz or 1) x key_len x key_len x d_k
            relations_keys = self.relative_positions_embeddings_k(
                relative_positions_matrix.to(key.device))
            #  (bsz or 1) x key_len x key_len x d_v
            relations_values = self.relative_positions_embeddings_v(
                relative_positions_matrix.to(key.device))

        # 2) Calculate and scale scores.
        # bsz x nhead x query_len x d_k
        query = query / math.sqrt(self.d_k)
        # batch x nhead x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0:
            if using_strct_relpos:
                assert relations_keys.dim() == 4
                # query_len x bsz x nhead x d_k
                permuted_query = query.permute(2, 0, 1, 3)
                # permuted_relations_keys: key_len x bsz x d_k x key_len
                permuted_relations_keys = relations_keys.permute(1, 0, 3, 2)
                # scores: key_len x bsz x nhead x key_len
                scores = torch.matmul(permuted_query, permuted_relations_keys)
                # scores: bsz x query_len x nhead x key_len
                scores = scores.permute(1, 2, 0, 3) + query_key
            else:
                scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key

        # we attend to every element in the key/value for a query
        scores = scores.float()  # bsz x nhead x query_len x key_len

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, query_len, key_len]
            scores = scores.masked_fill(mask, -1e18)

        if adj_mask is not None:
            assert adj_mask.size() == scores.size()
            scores = scores.masked_fill(~adj_mask.bool(), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        if adj_mask is not None:
            assert adj_mask.size() == attn.size()
            adj_mask = adj_mask.masked_fill(~adj_mask.bool(), 1e18)
            adj_mask = 1.0 / adj_mask
            attn = attn * adj_mask
            attn = f.normalize(attn, p=1, dim=-1)

        # bsz x nhead x query_len x key_len
        attn = self.dropout(attn)
        # bsz x nhead x query_len x d_v
        context_original = torch.matmul(attn, value)

        if self.max_relative_positions > 0:
            if using_strct_relpos:
                assert relations_values.dim() == 4
                # permuted_attn: query_len x bsz x nhead x key_len
                permuted_attn = attn.permute(2, 0, 1, 3)
                # relations_values: key_len x bsz x key_len x d_v
                add_term = torch.matmul(permuted_attn,
                                        relations_values.transpose(0, 1))
                # add_term: key_len x bsz x nhead x d_v
                add_term = add_term.permute(1, 2, 0, 3)
                context = unshape(context_original + add_term, self.d_v)
            else:
                context = unshape(context_original +
                                  relative_matmul(attn, relations_values, False),
                                  self.d_v)
        else:
            context = unshape(context_original, self.d_v)

        final_output = self.output(context)  # bsz x query_len x d_model

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]

        return final_output, attn_per_head
