import json
from udpipe import Model
from tqdm import tqdm

model_map = {
    'English': 'udpipe/english-ewt-ud-2.5-191206.udpipe',
    'Chinese': 'udpipe/chinese-gsd-ud-2.5-191206.udpipe',
    'Arabic': 'udpipe/arabic-padt-ud-2.5-191206.udpipe'
}


def get_conllu_text(sentence_index, model, key):
    sentences = []
    for sent in tqdm(sentence_index):
        orig_text = sent[key]
        text = orig_text.replace('<b>', '')
        text = text.replace('</b>', '')
        text = text.replace('<i>', '')
        text = text.replace('</i>', '')
        sentence = model.tokenize(text, 'ranges;presegmented')
        assert len(sentence) == 1
        sentence = sentence[0]
        sentence.setSentId(sent['id'])
        sentence.setText(orig_text)
        sentences.append(sentence)

    total_words = 0
    for s in sentences:
        total_words += len(s.words)
        model.tag(s)
        model.parse(s)
    conllu = model.write(sentences, "conllu")
    return conllu


def process(taskname, lang):
    with open('{}/{}/test_parallel.json'.format(taskname, lang)) as f:
        data = json.load(f)

    sentence_index = []
    for ex in data:
        sentence_index.append({
            'id': ex['id'],
            'source': ex['source'],
            'translation': ex['translation']
        })

    with open('{}/{}/sent_index.txt'.format(taskname, lang), 'w') as fw:
        for sent in sentence_index:
            fw.write(sent['id'] + '\n')

    model = Model(model_map['English'])
    with open('{}/{}/source_sent.conllu'.format(taskname, lang), 'w') as fw:
        src_conllu = get_conllu_text(sentence_index, model, 'source')
        fw.write(src_conllu)

    model = Model(model_map[lang])
    with open('{}/{}/parallel_sent.conllu'.format(taskname, lang), 'w') as fw:
        para_conllu = get_conllu_text(sentence_index, model, 'translation')
        fw.write(para_conllu)


if __name__ == '__main__':
    process('ace_event', 'Arabic')
    process('ace_relation', 'Arabic')
