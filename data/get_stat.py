import json
import os
from tqdm import tqdm
from collections import OrderedDict
from prettytable import PrettyTable
from clie.transformers import BertTokenizer
from clie.inputters.constant import MAX_BERT_SEQUENCE_LENGTH

ACE_KEYS = [
    'id',
    'token',
    'stanford_pos',
    'stanford_ner',
    'stanford_deprel',
    'relation',
    'stanford_head',
    'subj_type',
    'subj_start',
    'subj_end',
    'obj_type',
    'obj_start',
    'obj_end'
]


def main(dirpath, datatype, bert_tokenizer):
    records = {'train': 0, 'dev': 0, 'test': 0}
    max_record_length = {'train': 0, 'dev': 0, 'test': 0}
    max_bert_seq_length = {'train': 0, 'dev': 0, 'test': 0}
    unique_tokens = {'train': set(), 'dev': set(), 'test': set()}
    unique_pos_tokens = {'train': set(), 'dev': set(), 'test': set()}
    unique_ner_tokens = {'train': set(), 'dev': set(), 'test': set()}
    unique_deprel_labels = {'train': set(), 'dev': set(), 'test': set()}
    unique_subj_types = {'train': set(), 'dev': set(), 'test': set()}
    unique_obj_types = {'train': set(), 'dev': set(), 'test': set()}
    unique_labels = {'train': set(), 'dev': set(), 'test': set()}

    attribute_list = ["Records", "Max Record Length", "Max BERT Seq Length",
                      "Unique Tokens", "Unique POS Tokens", "Unique NER Tokens",
                      "Unique DepRel Labels", "Unique Subj Types",
                      "Unique Obj Types", "Unique Labels"]

    def read_data(split):
        with open(os.path.join(dirpath, '%s.json' % split)) as f:
            data = json.load(f)
            exception = 0
            for ex in tqdm(data, total=len(data)):
                records[split] += 1
                tokens = [tok.lower() for tok in ex['token']]
                if max_record_length[split] < len(tokens):
                    max_record_length[split] = len(tokens)
                bert_tokens = [piece for tok in ex['token']
                               for piece in bert_tokenizer.tokenize(tok)]
                if max_bert_seq_length[split] < len(bert_tokens):
                    max_bert_seq_length[split] = len(bert_tokens)
                if len(bert_tokens) > MAX_BERT_SEQUENCE_LENGTH - 2:
                    exception += 1

                unique_tokens[split].update(tokens)
                unique_pos_tokens[split].update(ex['stanford_pos'])
                unique_ner_tokens[split].update(ex['stanford_ner'])
                unique_ner_tokens[split].update(ex['stanford_ner'])
                unique_deprel_labels[split].update(ex['stanford_deprel'])
                unique_subj_types[split].update([ex['subj_type']])
                unique_obj_types[split].update([ex['obj_type']])
                unique_labels[split].update([ex['relation']])

        return exception

    oversized_bert_inputs = 0
    oversized_bert_inputs += read_data('train')
    oversized_bert_inputs += read_data('dev')
    oversized_bert_inputs += read_data('test')

    table = PrettyTable()
    table.field_names = [datatype, "Train", "Dev", "Test", "Fullset"]
    table.align[datatype] = "l"
    table.align["Train"] = "r"
    table.align["Valid"] = "r"
    table.align["Test"] = "r"
    table.align["Fullset"] = "r"

    fullset_stats = dict()
    for attr in attribute_list:
        var_name = '_'.join(attr.lower().split())
        var = eval(var_name)
        if isinstance(var['train'], set):
            val1 = len(var['train'])
            val2 = len(var['dev'])
            val3 = len(var['test'])
            fullset_stats[var_name] = var['train'].union(var['dev']).union(var['test'])
            fullset = len(fullset_stats[var_name])
        else:
            val1 = var['train']
            val2 = var['dev']
            val3 = var['test']
            if attr in ["Max Record Length", "Max BERT Seq Length"]:
                fullset = max([val1, val2, val3])
            else:
                fullset = val1 + val2 + val3

        if attr == 'Max BERT Seq Length':
            attr = 'Max Record Length (BERT)'
        table.add_row([attr, val1, val2, val3, fullset])

    print(table)
    if oversized_bert_inputs > 0:
        print('Warning - {} examples are longer than maximum length '
              'allowed for BERT'.format(oversized_bert_inputs))
    return fullset_stats


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer(
        os.path.join('bert_base_multilingual_cased', 'vocab.txt')
    )
    print('*' * 20 + ' English ' + '*' * 20)
    stat_en_event = main('ace_event/English/', 'Event', bert_tokenizer)
    stat_en_relation = main('ace_relation/English/', 'Relation', bert_tokenizer)
    print('*' * 20 + ' Arabic ' + '*' * 20)
    stat_ar_event = main('ace_event/Arabic/', 'Event', bert_tokenizer)
    stat_ar_relation = main('ace_relation/Arabic/', 'Relation', bert_tokenizer)
    print('*' * 20 + ' Chinese ' + '*' * 20)
    stat_zh_event = main('ace_event/Chinese/', 'Event', bert_tokenizer)
    stat_zh_relation = main('ace_relation/Chinese/', 'Relation', bert_tokenizer)

    with open('ace_event/vocab.txt', 'w') as fw:
        vocab = OrderedDict()
        for key in stat_en_event.keys():
            new_key = key.replace('unique_', '')
            if new_key == 'tokens':
                ar_tokens = ['!ar_{}'.format(w) for w in stat_ar_event[key]]
                en_tokens = ['!en_{}'.format(w) for w in stat_en_event[key]]
                zh_tokens = ['!zh_{}'.format(w) for w in stat_zh_event[key]]
                vocab[new_key] = ar_tokens + en_tokens + zh_tokens
            else:
                vocab[new_key] = list(stat_en_event[key].union(
                    stat_ar_event[key]).union(stat_zh_event[key]))
        json.dump(vocab, fw, indent=4, sort_keys=True)

    with open('ace_relation/vocab.txt', 'w') as fw:
        vocab = OrderedDict()
        for key in stat_en_relation.keys():
            new_key = key.replace('unique_', '')
            if new_key == 'tokens':
                ar_tokens = ['!ar_{}'.format(w) for w in stat_ar_relation[key]]
                en_tokens = ['!en_{}'.format(w) for w in stat_en_relation[key]]
                zh_tokens = ['!zh_{}'.format(w) for w in stat_zh_relation[key]]
                vocab[new_key] = ar_tokens + en_tokens + zh_tokens
            else:
                vocab[new_key] = list(stat_en_relation[key].union(
                    stat_ar_relation[key]).union(stat_zh_relation[key]))
        json.dump(vocab, fw, indent=4, sort_keys=True)
