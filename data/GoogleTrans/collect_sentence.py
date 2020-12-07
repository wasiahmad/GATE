import json


def get_sentences(filename):
    with open(filename) as f:
        data = json.load(f)

    sentences = []
    ignored_examples = 0
    total = 0
    for ex in data:
        if ex['subj_end'] - ex['subj_start'] < 0:
            ignored_examples += 1
            continue
        if ex['obj_end'] - ex['obj_start'] < 0:
            ignored_examples += 1
            continue

        text_tokens = []
        for i in range(len(ex['token'])):
            if i == ex['subj_start']:
                text_tokens.append('<b>')
            if i == ex['obj_start']:
                text_tokens.append('<i>')
            text_tokens.append(ex['token'][i])
            if i == ex['subj_end']:
                text_tokens.append('</b>')
            if i == ex['obj_end']:
                text_tokens.append('</i>')

        assert len(text_tokens) == len(ex['token']) + 4
        sentences.append({
            'id': ex['id'],
            'token': ex['token'],
            'sentence': ' '.join(text_tokens),
            'subj': [ex['subj_start'], ex['subj_end']],
            'obj': [ex['obj_start'], ex['obj_end']],
            'subj_type': ex['subj_type'],
            'obj_type': ex['obj_type'],
            'relation': ex['relation']
        })
        total += 1

    assert len(data) - ignored_examples == total
    print('Out of %d sentences, %d are ignored.' % (len(data), ignored_examples))
    return sentences


if __name__ == '__main__':
    event_en_test = get_sentences('../data/ace_event/English/test.json')
    with open('ace_event/en_test.json', 'w', encoding='utf-8') as fw:
        json.dump(event_en_test, fw, indent=4, sort_keys=True)
    relation_en_test = get_sentences('../data/ace_relation/English/test.json')
    with open('ace_relation/en_test.json', 'w', encoding='utf-8') as fw:
        json.dump(relation_en_test, fw, indent=4, sort_keys=True)
