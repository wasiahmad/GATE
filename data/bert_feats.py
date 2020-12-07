import json
import os
import torch
import subprocess
import numpy
import argparse

from tqdm import tqdm
from collections import namedtuple
from transformers import AutoTokenizer, AutoConfig, AutoModel

LANG_MAP = {
    'en': 'English',
    'ar': 'Arabic',
    'zh': 'Chinese'
}


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_example_to_features(ex_id,
                                tokens,
                                tokenizer,
                                max_seq_length,
                                doc_stride):
    assert isinstance(tokens, list)

    all_doc_tokens = []
    tok_to_orig_index = []
    unk_token = 0
    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.tokenize(token)
        # NOTE: In some cases, tokenizer doesn't output sub_tokens
        # e.g., '\xad'
        if len(sub_tokens) == 0:
            sub_tokens = [tokenizer.unk_token]
            unk_token += 1
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if unk_token > 0:
        print('Warning: [ex_id: %s] %d/%d words are replaced by special token %s.' %
              (ex_id, unk_token, len(tokens), tokenizer.unk_token))

    # The -2 accounts for [CLS] and [SEP]
    max_tokens_for_doc = max_seq_length - 2

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    features = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                                  split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])

        input_ids = tokenizer.convert_tokens_to_ids(
            [tokenizer.cls_token] + tokens + [tokenizer.sep_token])

        assert len(tokens) == len(token_to_orig_map)
        features.append({
            'input_ids': input_ids,
            'tokens': tokens,
            'token_to_orig_map': token_to_orig_map,
            'token_is_max_context': token_is_max_context,
        })

    return features


def main(src_dir, tgt_dir, bert_model, lang):
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    config = AutoConfig.from_pretrained(bert_model)
    encoder = AutoModel.from_pretrained(bert_model, config=config)
    encoder.cuda()
    encoder.eval()

    max_tokens_for_doc = 256
    doc_stride = 128

    for split in ['train', 'dev', 'test']:
        with open(os.path.join(src_dir, '%s.json' % split)) as f:
            data = json.load(f)
            train_context_embeddings = {}
            for ex_idx, ex in enumerate(tqdm(data, total=len(data))):
                ex_id = '{}_{}'.format(ex_idx, ex['id'])
                features = convert_example_to_features(ex_id,
                                                       ex['token'],
                                                       tokenizer,
                                                       max_tokens_for_doc,
                                                       doc_stride)
                assert len(features) > 0
                context_embeds = [numpy.zeros((config.hidden_size))] * len(ex['token'])

                vec_counts = {}
                for feat in features:
                    ids_tensor = torch.LongTensor(feat['input_ids']). \
                        unsqueeze(0).cuda(non_blocking=True)
                    with torch.no_grad():
                        sequence_output = encoder(input_ids=ids_tensor)[0].squeeze(0)
                    sequence_output = sequence_output[1:-1].cpu().numpy()  # removing [CLS] and [SEP]
                    assert len(feat['tokens']) == sequence_output.shape[0]
                    for idx in range(sequence_output.shape[0]):
                        if feat['token_is_max_context'][idx]:
                            orig_idx = feat['token_to_orig_map'][idx]
                            if orig_idx not in vec_counts:
                                vec_counts[orig_idx] = 0
                            vec_counts[orig_idx] += 1
                            context_embeds[orig_idx] = context_embeds[orig_idx] + sequence_output[idx]

                if len(vec_counts) != len(context_embeds):
                    raise ValueError('{} != {}'.format(len(vec_counts), len(context_embeds)))
                # we take the average of embeddings for each word piece
                for idx in range(len(context_embeds)):
                    context_embeds[idx] = numpy.divide(context_embeds[idx], vec_counts[idx])
                assert ex_id not in train_context_embeddings
                train_context_embeddings[ex_id] = numpy.stack(context_embeds, axis=0)

            assert len(train_context_embeddings) == len(data), \
                '{} != {}'.format(len(train_context_embeddings), len(data))
            numpy.savez(os.path.join(tgt_dir, '%s.npz' % split), **train_context_embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help="Path of source directory")
    parser.add_argument('--tgt_dir', type=str, required=True, help="Path of target directory")
    parser.add_argument('--bert_model', type=str, required=True, help="BERT model name")
    parser.add_argument('--lang', type=str, help="Name of the language", default='en',
                        choices=['en', 'ar', 'zh'])
    args = parser.parse_args()
    args.src_dir = os.path.join(args.src_dir, LANG_MAP[args.lang])
    args.tgt_dir = os.path.join(args.tgt_dir, LANG_MAP[args.lang])
    if not os.path.exists(args.src_dir):
        print("Directory {} does not exist.".format(args.src_dir))
        raise FileNotFoundError
    if not os.path.exists(args.tgt_dir):
        os.makedirs(args.tgt_dir)
        print("{} directory created.".format(args.tgt_dir))

    main(args.src_dir, args.tgt_dir, args.bert_model, args.lang)
