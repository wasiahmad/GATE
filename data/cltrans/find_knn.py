import io
import json
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluation of word alignment')
parser.add_argument("--src_vocab", type=str, required=True, help="Vocab file")
parser.add_argument("--tgt_file", type=str, required=True, help="Target filename")
parser.add_argument("--src_emb", type=str, required=True, help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, required=True, help="Load target embeddings")
parser.add_argument("--src_lang", type=str, required=True, help="Source language")
parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--maxload", type=int, default=200000)
parser.add_argument("--knn", type=int, default=3)
params = parser.parse_args()


def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    word2id, id2word = {}, {}
    for i, line in tqdm(enumerate(fin), total=n):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        id2word[i] = tokens[0]
        word2id[tokens[0]] = i
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded from %s" % (len(word2id), fname))
    return word2id, id2word, x


def compute_csls_nn(x_src, x_tgt, k=10, bsz=1024):
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sc = np.dot(x_src, x_tgt.T)  # cosine-sim
    similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])
    for i in tqdm(range(0, x_tgt.shape[0], bsz)):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    similarities -= sc2[np.newaxis, :]

    nn = np.argsort(-similarities, axis=-1)[:, :k].tolist()
    return nn


with open(params.src_vocab) as f:
    vocab = json.load(f)
    src_vocab = []
    for w in vocab['tokens']:
        if w.startswith('!{}_'.format(params.src_lang)):
            w = w.replace('!{}_'.format(params.src_lang), '')
            src_vocab.append(w)

w2id_src, id2w_src, x_src = load_vectors(params.src_emb,
                                         maxload=params.maxload,
                                         center=params.center)
w2id_tgt, id2w_tgt, x_tgt = load_vectors(params.tgt_emb,
                                         maxload=params.maxload,
                                         center=params.center)
vocab_embed = []
words_found = []
for idx, w in enumerate(src_vocab):
    if w in w2id_src:
        index = w2id_src[w]
        vocab_embed.append(x_src[index])
        words_found.append(w)

vocab_embed = np.stack(vocab_embed, axis=0)
print('Total word found - ', vocab_embed.shape[0])

print('Computing K-nearest neighbors ...')
knn = compute_csls_nn(vocab_embed, x_tgt, k=params.knn, bsz=1024)

nearest_neighbors = {}
for idx, (src_word, tgt_indices) in enumerate(zip(words_found, knn)):
    assert len(tgt_indices) == params.knn, len(tgt_indices)
    src_word = '!{}_{}'.format(params.src_lang, src_word)
    nearest_neighbors[src_word] = []
    for nn in tgt_indices:
        nn_word = '!{}_{}'.format(params.tgt_lang, id2w_tgt[nn])
        nearest_neighbors[src_word].append(nn_word)

with open(params.tgt_file, 'w') as fw:
    json.dump(nearest_neighbors, fw, indent=4, sort_keys=True)
