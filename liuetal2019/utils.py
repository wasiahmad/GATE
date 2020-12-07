import io
import logging
import json
import numpy
import torch
import numpy as np
from tqdm import tqdm
from clie.inputters import constant
from clie.objects import Sentence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

logger = logging.getLogger(__name__)


def load_word_embeddings(file):
    embeddings_index = {}
    fin = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    for i, line in tqdm(enumerate(fin), total=n):
        tokens = line.rstrip().split(' ')
        v = numpy.array(tokens[1:], dtype=float)
        embeddings_index[tokens[0]] = v

    return embeddings_index


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(filename, src_lang, tgt_lang, knn_file,
              knn_size, max_examples=-1):
    examples = []
    wrong_subj_pos, wrong_obj_pos = 0, 0
    with open(filename) as f:
        data = json.load(f)
        knn_dict = None
        if knn_file:
            with open(knn_file) as f:
                knn_dict = json.load(f)
        for idx, ex in enumerate(tqdm(data, total=len(data))):
            sentence = Sentence(ex['id'])
            sentence.language = src_lang
            sentence.words = ex['token']
            sentence.pos = ex['stanford_pos']
            sentence.ner = ex['stanford_ner']
            sentence.deprel = ex['stanford_deprel']
            sentence.head = [int(x) for x in ex['stanford_head']]
            sentence.subj_type = ex['subj_type']
            sentence.obj_type = ex['obj_type']
            sentence.relation = ex['relation']

            if ex['subj_end'] - ex['subj_start'] < 0:
                # we swap the start and end index
                wrong_subj_pos += 1
                sentence.subject = [ex['subj_end'], ex['subj_start']]
            else:
                sentence.subject = [ex['subj_start'], ex['subj_end']]

            if ex['obj_end'] - ex['obj_start'] < 0:
                # we swap the start and end index
                wrong_obj_pos += 1
                sentence.object = [ex['obj_end'], ex['obj_start']]
            else:
                sentence.object = [ex['obj_start'], ex['obj_end']]

            # store KNN word info
            if knn_dict:
                sentence.tgt_lang = tgt_lang
                knn_words = []
                for w in ex['token']:
                    w = '!{}_{}'.format(src_lang, w)
                    if w in knn_dict:
                        assert len(knn_dict[w]) == knn_size
                        knn_words.append(knn_dict[w])
                    else:
                        knn_words.append([constant.UNK_WORD] * knn_size)
                sentence.knn_words = knn_words

            examples.append(sentence)
            if max_examples != -1 and len(examples) > max_examples:
                break

    if wrong_subj_pos > 0 or wrong_obj_pos > 0:
        logger.info('{} and {} wrong subject and object positions found!'.format(
            wrong_subj_pos, wrong_obj_pos))

    return examples


def vectorize(ex, model, iseval):
    """Torchify a single example."""

    words = ['!{}_{}'.format(ex.language, w) for w in ex.words]
    words = [model.word_dict[w] for w in words]
    knn_word = None
    if ex.knn_words:
        knn_word = [[model.word_dict[w] for w in knn]
                    for knn in ex.knn_words]
        knn_word = torch.LongTensor(knn_word)

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

    return {
        'id': ex.id,
        'language': ex.language,
        'word': word,
        'pos': pos,
        'ner': ner,
        'deprel': deprel,
        'type': type,
        'head': head,
        'subject': ex.subj_text,
        'object': ex.obj_text,
        'subject_pos': subj_position,
        'object_pos': obj_position,
        'relation': model.label_dict[ex.relation],
        'knn_word': knn_word
    }


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]
    use_knn = batch[0]['knn_word'] is not None
    # NOTE. batch[0]['knn_word'] is a 2d list
    knn_size = len(batch[0]['knn_word'][0]) if use_knn else 0

    # --------- Prepare Code tensors ---------
    max_len = max([ex['word'].size(0) for ex in batch])

    # Batch Code Representations
    len_rep = torch.LongTensor(batch_size).fill_(constant.PAD)
    word_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    head_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    subject_pos_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    object_pos_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    pos_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    ner_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    deprel_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    type_rep = torch.LongTensor(batch_size, max_len).fill_(constant.PAD)
    labels = torch.LongTensor(batch_size)
    subject = []
    object = []

    knn_rep = None
    if use_knn:
        knn_rep = torch.LongTensor(batch_size, max_len, knn_size).fill_(constant.PAD)

    for i, ex in enumerate(batch):
        len_rep[i] = ex['word'].size(0)
        labels[i] = ex['relation']
        word_rep[i, :len_rep[i]] = ex['word']
        head_rep[i, :len_rep[i]] = ex['head']
        subject_pos_rep[i, :len_rep[i]] = ex['subject_pos']
        object_pos_rep[i, :len_rep[i]] = ex['object_pos']
        pos_rep[i, :len_rep[i]] = ex['pos']
        ner_rep[i, :len_rep[i]] = ex['ner']
        deprel_rep[i, :len_rep[i]] = ex['deprel']
        type_rep[i, :len_rep[i]] = ex['type']
        subject.append(ex['subject'])
        object.append(ex['object'])
        if use_knn:
            knn_rep[i, :len_rep[i]] = ex['knn_word']

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'len_rep': len_rep,
        'word_rep': word_rep,
        'knn_rep': knn_rep,
        'head_rep': head_rep,
        'subject': subject,
        'object': object,
        'subject_pos_rep': subject_pos_rep,
        'object_pos_rep': object_pos_rep,
        'labels': labels,
        'pos_rep': pos_rep,
        'ner_rep': ner_rep,
        'deprel_rep': deprel_rep,
        'type_rep': type_rep
    }


class ACE05Dataset(Dataset):
    def __init__(self, examples, model, evaluation=False):
        self.model = model
        self.examples = examples
        self.evaluation = evaluation

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model,
                         iseval=self.evaluation)

    def lengths(self):
        return [len(ex.words) for ex in self.examples]


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
