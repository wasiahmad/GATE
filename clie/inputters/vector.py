import torch
import numpy
from clie.inputters import constant
from clie.objects import tree_to_adj, tree_to_dist_mat


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK if x != constant.UNK and numpy.random.random() < dropout \
                else x for x in tokens]


def vectorize(ex, model, iseval):
    """Torchify a single example."""

    words = ['!{}_{}'.format(ex.language, w) for w in ex.words]
    words = [model.word_dict[w] for w in words]
    if iseval:
        words = word_dropout(words, model.args.word_dropout)

    word = torch.LongTensor(words)
    pos = torch.LongTensor([model.pos_dict[p] for p in ex.pos])
    ner = torch.LongTensor([model.ner_dict[n] for n in ex.ner])
    deprel = torch.LongTensor([model.deprel_dict[d] for d in ex.deprel])
    assert any([x == 0 for x in ex.head])
    head = torch.LongTensor(ex.head)
    subj_position = torch.LongTensor(ex.subj_position)
    obj_position = torch.LongTensor(ex.obj_position)

    type = [0] * len(ex.words)
    ttype = model.type_dict[ex.subj_type]
    start, end = ex.subject
    type[start: end + 1] = [ttype] * (end - start + 1)
    atype = model.type_dict[ex.obj_type]
    start, end = ex.object
    type[start: end + 1] = [atype] * (end - start + 1)
    type = torch.LongTensor(type)

    bert_embeds = None
    if ex.bert_vectors is not None:
        bert_embeds = torch.from_numpy(ex.bert_vectors)

    position_rep = None
    root = None
    if model.args.struct_position:
        # ex.struct_position is absolute position representation
        # that considers the dependency structure
        root, pos_rep = ex.struct_position
        position_rep = torch.LongTensor(pos_rep)

    use_relative_pos = False
    max_pos = model.args.max_relative_pos
    if isinstance(max_pos, int) and max_pos > 0:
        use_relative_pos = True
    if isinstance(max_pos, list) and any(max_pos):
        use_relative_pos = True

    adj_mask = None
    if model.args.embed_graph > 0:
        adj_mask = ex.dist_mat(prune=model.args.prune_k, self_loop=True)
        # if model.args.max_tree_dist > 1:
        #     adj_mask = ex.dist_mat(prune=model.args.prune_k, self_loop=True)
        # else:
        #     adj_mask = ex.adj_mat(prune=model.args.prune_k, self_loop=True)

    return {
        'id': ex.id,
        'language': ex.language,
        'word': word,
        'pos': pos,
        'ner': ner,
        'deprel': deprel,
        'head': head,
        'type': type,
        'subject': ex.subj_text,
        'object': ex.obj_text,
        'subject_pos': subj_position,
        'object_pos': obj_position,
        'position_rep': position_rep,
        'relation': model.label_dict[ex.relation],
        'bert_embeds': bert_embeds,
        'max_seq_len': model.args.max_src_len,
        'root': root,
        'use_relative_pos': use_relative_pos,
        'adj_mask': adj_mask
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]
    use_bert = batch[0]['bert_embeds'] is not None
    use_position = batch[0]['position_rep'] is not None
    max_seq_len = batch[0]['max_seq_len']
    compute_adj_mask = batch[0]['adj_mask'] is not None
    # to use structural relative position representations,
    # we need to compute an `n x n` matrix for attention computation
    struct_relative_pos = False
    if use_position and batch[0]['use_relative_pos']:
        struct_relative_pos = True

    # --------- Prepare Code tensors ---------
    max_len = max([ex['word'].size(0) for ex in batch])

    # Batch Code Representations
    len_rep = torch.LongTensor(batch_size).fill_(constant.PAD)
    word_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    pos_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    ner_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    deprel_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    head_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    type_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    subject_pos_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    object_pos_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    labels = torch.LongTensor(batch_size)
    subject = []
    object = []

    adj_mask = None
    if compute_adj_mask:
        adj_mask = torch.FloatTensor(batch_size, max_len, max_len).zero_()

    position_rep = None
    if use_position:
        position_rep = torch.LongTensor(batch_size, max_len).fill_(max_seq_len)

    struct_relpos_rep = None
    template_mat = None
    if struct_relative_pos:
        struct_relpos_rep = torch.LongTensor(batch_size, max_len, max_len).fill_(max_seq_len)
        template_mat = torch.zeros((max_len, max_len))
        [rows, cols] = torch.triu_indices(max_len, max_len)
        template_mat[rows, cols] = 1
        [rows, cols] = torch.tril_indices(max_len, max_len)
        template_mat[rows, cols] = -1
        template_mat.fill_diagonal_(0)

    bert_embed_rep = None
    if use_bert:
        bert_max_length = max([ex['bert_embeds'].shape[0] for ex in batch])
        assert bert_max_length == max_len
        embed_size = batch[0]['bert_embeds'].shape[1]
        bert_embed_rep = torch.FloatTensor(batch_size, bert_max_length, embed_size).zero_()

    for i, ex in enumerate(batch):
        len_rep[i] = ex['word'].size(0)
        labels[i] = ex['relation']
        word_rep[i, :len_rep[i]] = ex['word']
        pos_rep[i, :len_rep[i]] = ex['pos']
        ner_rep[i, :len_rep[i]] = ex['ner']
        deprel_rep[i, :len_rep[i]] = ex['deprel']
        head_rep[i, :len_rep[i]] = ex['head']
        type_rep[i, :len_rep[i]] = ex['type']
        subject_pos_rep[i, :len_rep[i]] = ex['subject_pos']
        object_pos_rep[i, :len_rep[i]] = ex['object_pos']
        subject.append(ex['subject'])
        object.append(ex['object'])
        if compute_adj_mask:
            adj_mask[i, :len_rep[i], :len_rep[i]] = torch.from_numpy(ex['adj_mask'])
        if use_position:
            position_rep[i, :len_rep[i]] = ex['position_rep']
        if use_bert:
            bert_embed_rep[i, :len_rep[i]] = ex['bert_embeds']
        if struct_relative_pos:
            dist_matrix = tree_to_dist_mat(max_len, ex['root'], directed=False)
            dist_matrix = torch.from_numpy(dist_matrix)
            dist_matrix = dist_matrix * template_mat
            struct_relpos_rep[i] = (dist_matrix * template_mat).long()

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'len_rep': len_rep,
        'word_rep': word_rep,
        'position_rep': position_rep,
        'pos_rep': pos_rep,
        'ner_rep': ner_rep,
        'deprel_rep': deprel_rep,
        'head_rep': head_rep,
        'type_rep': type_rep,
        'subject': subject,
        'object': object,
        'subject_pos_rep': subject_pos_rep,
        'object_pos_rep': object_pos_rep,
        'labels': labels,
        'bert_embeds': bert_embed_rep,
        'adj_mask': adj_mask,
        'struct_relpos_rep': struct_relpos_rep
    }
