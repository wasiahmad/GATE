import io
import logging
import json
import numpy
from tqdm import tqdm
from clie.objects import Sentence

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


def load_data(filename,
              language,
              bert_feat_file,
              max_examples=-1):
    examples = []
    wrong_subj_pos, wrong_obj_pos = 0, 0
    with open(filename) as f:
        data = json.load(f)
        bert_feats = None
        if bert_feat_file:
            bert_feats = numpy.load(bert_feat_file)
            logger.info('BERT features are loaded from %s', bert_feat_file)
        for idx, ex in enumerate(tqdm(data, total=len(data))):
            sentence = Sentence(ex['id'])
            sentence.language = language
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

            if bert_feats is not None:
                ex_id = '{}_{}'.format(idx, ex['id'])
                sentence.bert_vectors = bert_feats[ex_id]
            examples.append(sentence)

            if max_examples != -1 and len(examples) > max_examples:
                break

    if wrong_subj_pos > 0 or wrong_obj_pos > 0:
        logger.info('{} and {} wrong subject and object positions found!'.format(
            wrong_subj_pos, wrong_obj_pos))

    return examples
