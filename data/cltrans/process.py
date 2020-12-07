import json
import os
import io
import numpy
from tqdm import tqdm
import argparse

LANG_MAP = {
    'en': 'English',
    'ar': 'Arabic',
    'zh': 'Chinese'
}


def guess_language_id(file_path):
    bname = os.path.basename(file_path)
    lang_id = bname.lower().split('.')[1]
    assert lang_id in LANG_MAP.keys(), \
        "Unknown lang id %s from path %s" % (lang_id, file_path)
    return lang_id


def load_word_embeddings(file):
    embeddings_index = {}
    fin = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    print('Loading embeddings of {} words from {}.'.format(n, file))
    for i, line in tqdm(enumerate(fin), total=n):
        tokens = line.rstrip().split(' ')
        v = numpy.array(tokens[1:], dtype=float)
        embeddings_index[tokens[0]] = v

    return embeddings_index


def save_word_embeddings(file, embeddings_index, emb_size):
    n = len(embeddings_index)
    print('Saving embeddings ({}d) for {} words.'.format(emb_size, n))
    fout = io.open(file, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, emb_size))
    for word, vec in tqdm(embeddings_index.items(), total=n):
        assert vec.shape[0] == emb_size, '{} != {}'.format(vec.shape[0], emb_size)
        fout.write(word + " " + " ".join(map(lambda a: "%.4f" % a, vec)) + "\n")
    fout.close()


def identify_language(token):
    if token.startswith("!en_"):
        return 'en', token[4:]
    elif token.startswith("!ar_"):
        return 'ar', token[4:]
    elif token.startswith("!zh_"):
        return 'zh', token[4:]
    else:
        raise ValueError


def filter_embeddings(args, embfiles):
    vocab_file, languages = args.vocab_file, args.lang
    tgt_emb_file = os.path.join(args.tgt_dir, args.tgt_file)
    tgt_vocab_file = os.path.join(args.tgt_dir, 'vocab.txt')

    lang_ids = [guess_language_id(f) for f in embfiles]
    assert lang_ids == languages

    with open(vocab_file) as f:
        vocab = json.load(f)
        unique_tokens = set(vocab['tokens'])

    # read knn files to update vocab
    for src_lang in args.lang:
        for tgt_lang in args.lang:
            filename = '{}_{}_knn.txt'.format(src_lang, tgt_lang)
            filepath = os.path.join(args.tgt_dir, filename)
            if os.path.exists(filepath):
                with open(filepath) as f:
                    knn_vocab = json.load(f)
                    for src_word, knn_words in knn_vocab.items():
                        for word in knn_words:
                            unique_tokens.add(word)

    # update and save vocab with nearest neighbor words
    vocab['tokens'] = [w for w in sorted(list(unique_tokens))]
    with open(tgt_vocab_file, 'w') as fw:
        json.dump(vocab, fw, indent=4, sort_keys=True)

    emb_indices = {lang: load_word_embeddings(file)
                   for lang, file in zip(languages, embfiles)}
    assert len(emb_indices) == len(languages)

    filtered_embeddings = dict()
    for lang_spec_token in unique_tokens:
        lang_id, token = identify_language(lang_spec_token)
        if token in emb_indices[lang_id]:
            filtered_embeddings[lang_spec_token] = emb_indices[lang_id][token]
        elif token.lower() in emb_indices[lang_id]:
            filtered_embeddings[lang_spec_token] = emb_indices[lang_id][token.lower()]

    save_word_embeddings(tgt_emb_file, filtered_embeddings, 300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help="Path of source embeddings directory")
    parser.add_argument('--tgt_dir', type=str, required=True, help="Path of target embeddings directory")
    parser.add_argument('--src_file', type=str, nargs='+', required=True, help="Source embedding file name")
    parser.add_argument('--tgt_file', type=str, required=True, help="Target embedding file name")
    parser.add_argument('--lang', type=str, nargs='+', help="Name of the languages", required=True)
    parser.add_argument('--vocab_file', type=str, help="Vocabulary file", required=True)
    args = parser.parse_args()

    assert len(args.lang) == len(args.src_file)

    embed_files = []
    for srcfile in args.src_file:
        embed_file = os.path.join(args.src_dir, srcfile)
        assert os.path.exists(embed_file)
        embed_files.append(embed_file)

    assert os.path.exists(args.vocab_file)
    filter_embeddings(args, embed_files)
