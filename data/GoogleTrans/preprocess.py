import json
import os
from udpipe import Model
from conllu import parse
from collections import OrderedDict

model_map = {
    'en': 'udpipe/english-ewt-ud-2.5-191206.udpipe',
    'zh': 'udpipe/chinese-gsd-ud-2.5-191206.udpipe',
    'ar': 'udpipe/arabic-padt-ud-2.5-191206.udpipe'
}


def find_span(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    if len(start) == 0:
        start = [i for i, tok in enumerate(offsets) if tok[0] < begin_offset < tok[1]]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    if len(end) == 0:
        end = [i for i, tok in enumerate(offsets) if tok[0] < end_offset < tok[1]]
    assert (len(start) <= 1)
    assert (len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]

    return False


def load_conllu(conllu_text):
    conllu_data = []
    sentences = parse(conllu_text)
    for idx, sentence in enumerate(sentences):
        tokens, upos, head, deprel, offset = [], [], [], [], []
        reserved_offsets = []
        for widx, word in enumerate(sentence):
            if isinstance(word['id'], tuple):
                # multi-word token, e.g., word['id'] = (4, '-', 5)
                assert len(word['id']) == 3
                indices = word['misc']['TokenRange'].split(':')
                reserved_offsets.append([int(indices[0]), int(indices[1])])
            else:
                tokens.append(word['form'])
                upos.append(word['upostag'])
                head.append(word['head'])
                deprel.append(word['deprel'])
                if word['misc'] is not None:
                    # single-word token
                    indices = word['misc']['TokenRange'].split(':')
                    offset.append([int(indices[0]), int(indices[1])])
                elif len(reserved_offsets) > 0:
                    offset.append(reserved_offsets.pop())
                else:
                    offset.append([-1, -1])

        assert len(tokens) == len(offset)
        sent_obj = OrderedDict([
            ('token', tokens),
            ('stanford_pos', upos),
            ('stanford_head', head),
            ('stanford_deprel', deprel),
            ('offset', offset)
        ])
        conllu_data.append(sent_obj)

    return conllu_data


def preprocess(srcfile):
    with open(srcfile) as f:
        data = json.load(f)

    confusing = 0
    returned_data = []

    for ex in data:
        if ex['parallel'].count('<b>') > 1:
            confusing += 1
            continue
        elif ex['parallel'].count('</b>') > 1:
            confusing += 1
            continue
        elif ex['parallel'].count('<i>') > 1:
            confusing += 1
            continue
        elif ex['parallel'].count('</i>') > 1:
            confusing += 1
            continue

        parallel_sent = ex['parallel']
        subj_start = parallel_sent.find('<b>')
        subj_end = parallel_sent.find('</b>')
        obj_start = parallel_sent.find('<i>')
        obj_end = parallel_sent.find('</i>')

        if subj_start > subj_end:
            confusing += 1
            continue
        elif obj_start > obj_end:
            confusing += 1
            continue

        if subj_end < obj_start:
            # subj is in the left of obj
            position = ['subj_start', 'subj_end', 'obj_start', 'obj_end']
        elif obj_end < subj_start:
            # obj is in the left of subj
            position = ['obj_start', 'obj_end', 'subj_start', 'subj_end']
        elif subj_start < obj_start < subj_end:
            position = ['subj_start', 'obj_start']
            if obj_end < subj_end:
                position += ['obj_end', 'subj_end']
            else:
                position += ['subj_end', 'obj_end']
        elif obj_start < subj_start < obj_end:
            position = ['obj_start', 'subj_start']
            if subj_end < obj_end:
                position += ['subj_end', 'obj_end']
            else:
                position += ['obj_end', 'subj_end']
        else:
            raise ValueError()

        # print(parallel_sent)
        for item in position:
            if item == 'subj_start':
                real_ss = parallel_sent.find('<b>')
                parallel_sent = parallel_sent.replace('<b>', '')
            elif item == 'subj_end':
                real_se = parallel_sent.find('</b>')
                parallel_sent = parallel_sent.replace('</b>', '')
            elif item == 'obj_start':
                real_os = parallel_sent.find('<i>')
                parallel_sent = parallel_sent.replace('<i>', '')
            elif item == 'obj_end':
                real_oe = parallel_sent.find('</i>')
                parallel_sent = parallel_sent.replace('</i>', '')

        if real_ss == real_se:
            confusing += 1
            continue
        elif real_os == real_oe:
            confusing += 1
            continue

        ex['parallel'] = {
            'sentence': parallel_sent,
            'subj_pos': [real_ss, real_se],
            'obj_pos': [real_os, real_oe],
            'source': ex['parallel']
        }
        returned_data.append(ex)

    #     print(parallel_sent)
    #     print(position)
    #     print(real_ss, real_se, real_os, real_oe)
    #     print()

    print('Out of %d examples, %d are dropped!' % (len(data), confusing))
    return returned_data


def get_conllu_text(text, model):
    sentences = model.tokenize(text, 'ranges;presegmented')
    total_words = 0
    for s in sentences:
        total_words += len(s.words)
        model.tag(s)
        model.parse(s)
    conllu = model.write(sentences, "conllu")
    return conllu


def convert_char_to_word_indices(parallel_data, tgtfile, lang):
    model = Model(model_map[lang])
    skipped = 0
    for ex in parallel_data:
        trans_sent = ex['parallel']['sentence']
        conllu_text = get_conllu_text(trans_sent, model)
        conll_ex = load_conllu(conllu_text)
        assert len(conll_ex) == 1
        conll_ex = conll_ex[0]

        subj_start_char, subj_end_char = ex['parallel']['subj_pos']
        obj_start_char, obj_end_char = ex['parallel']['obj_pos']
        subj_start_end = find_span(conll_ex['offset'], subj_start_char, subj_end_char)
        obj_start_end = find_span(conll_ex['offset'], obj_start_char, obj_end_char)

        if not subj_start_end:
            print(conll_ex['token'])
            print(conll_ex['offset'])
            print(trans_sent[subj_start_char:subj_end_char])
            print(subj_start_char, subj_end_char)
            skipped += 1
            continue
        if not obj_start_end:
            print(conll_ex['token'])
            print(conll_ex['offset'])
            print(trans_sent[obj_start_char:obj_end_char])
            print(obj_start_char, obj_end_char)
            skipped += 1
            continue

        ex['source'] = ex['sentence']
        ex['parallel'].pop('subj_pos')
        ex['parallel'].pop('obj_pos')
        ex.pop('sentence')
        ex.pop('subj')
        ex.pop('obj')
        ex.pop('token')
        ex['translation'] = ex['parallel']['source']
        ex.pop('parallel')

        ex['token'] = conll_ex['token']
        ex['stanford_pos'] = conll_ex['stanford_pos']
        ex['stanford_head'] = conll_ex['stanford_head']
        ex['stanford_deprel'] = conll_ex['stanford_deprel']
        ex['stanford_ner'] = ["O"] * len(ex['token'])
        ex['subj_start'] = subj_start_end[0]
        ex['subj_end'] = subj_start_end[1]
        ex['obj_start'] = obj_start_end[0]
        ex['obj_end'] = obj_start_end[1]
        ex['subj_type'] = ex['subj_type']
        ex['obj_type'] = ex['obj_type']
        ex['relation'] = ex['relation']

    if skipped > 0:
        print('%d examples are skipped since we cannot resolve their character indices.' % skipped)
    with open(tgtfile, 'w') as fw:
        json.dump(parallel_data, fw, sort_keys=True, indent=4)


def filter_source_examples(selected_ids, src_file, tgt_file):
    selected_data = []
    with open(src_file) as f:
        data = json.load(f)

    data_dict = {ex['id']: ex for ex in data}
    for idx in selected_ids:
        selected_data.append(data_dict[idx])

    assert len(selected_data) == len(selected_ids), \
        '{} != {}'.format(len(selected_data), len(selected_ids))
    with open(tgt_file, 'w') as fw:
        json.dump(selected_data, fw, sort_keys=True, indent=4)


if __name__ == '__main__':
    new_data = preprocess('ace_event/en_test_zh.json')
    convert_char_to_word_indices(new_data,
                                 'ace_event/Chinese/test_parallel.json',
                                 'zh')
    selected_ids = [ex['id'] for ex in new_data]
    filter_source_examples(selected_ids,
                           '../data/ace_event/English/test.json',
                           'ace_event/Chinese/test_source.json')

    new_data = preprocess('ace_relation/en_test_zh.json')
    convert_char_to_word_indices(new_data,
                                 'ace_relation/Chinese/test_parallel.json',
                                 'zh')
    selected_ids = [ex['id'] for ex in new_data]
    filter_source_examples(selected_ids,
                           '../data/ace_relation/English/test.json',
                           'ace_relation/Chinese/test_source.json')

    new_data = preprocess('ace_event/en_test_ar.json')
    convert_char_to_word_indices(new_data,
                                 'ace_event/Arabic/test_parallel.json',
                                 'ar')
    selected_ids = [ex['id'] for ex in new_data]
    filter_source_examples(selected_ids,
                           '../data/ace_event/English/test.json',
                           'ace_event/Arabic/test_source.json')

    new_data = preprocess('ace_relation/en_test_ar.json')
    convert_char_to_word_indices(new_data,
                                 'ace_relation/Arabic/test_parallel.json',
                                 'ar')
    selected_ids = [ex['id'] for ex in new_data]
    filter_source_examples(selected_ids,
                           '../data/ace_relation/English/test.json',
                           'ace_relation/Arabic/test_source.json')
