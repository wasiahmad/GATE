import re
import os
import argparse
from prettytable import PrettyTable

LANG_MAP = {
    'en': 'English',
    'ar': 'Arabic',
    'zh': 'Chinese'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def single_source(opt):
    result = {}
    pattern = 'precision = (\\d+\\.\\d+) \\| recall = (\\d+\\.\\d+) \\| f1 = (\\d+\\.\\d+)'
    for src_lang in ['en', 'ar', 'zh']:
        for tgt_lang in ['en', 'ar', 'zh']:
            filename = os.path.join(opt.dir, '{}_{}_{}_test.txt'.format(
                src_lang, opt.model_name, tgt_lang))
            if os.path.isfile(filename):
                with open(filename) as f:
                    for line in f:
                        line = line.strip()
                        out = re.findall(pattern, line)
                        if out:
                            assert len(out) == 1 and len(out[0]) == 3
                            result['{}_{}'.format(src_lang, tgt_lang)] = [float(v) for v in out[0]]
                            break

    table = PrettyTable()
    table.field_names = ["", "English", "Chinese", "Arabic", "Average"]
    table.align[""] = "l"
    table.align["English"] = "c"
    table.align["Chinese"] = "c"
    table.align["Arabic"] = "c"
    for src_lang in ['en', 'zh', 'ar']:
        f1_scores = []
        scores = []
        for tgt_lang in ['en', 'zh', 'ar']:
            label = '{}_{}'.format(src_lang, tgt_lang)
            if label in result:
                f1_scores.append(result[label][2])
                scores.append('/'.join(['{:0.2f}'.format(v) for v in result[label]]))
            else:
                scores.append('NA')

        average = '{:0.2f}'.format(sum(f1_scores) / len(f1_scores)) \
            if f1_scores else 'NA'
        table.add_row([LANG_MAP[src_lang], *scores, average])
    print(table)


def multi_source(opt):
    result = {}
    pattern = 'precision = (\\d+\\.\\d+) \\| recall = (\\d+\\.\\d+) \\| f1 = (\\d+\\.\\d+)'
    src_langs = ["en_zh", "en_ar", "ar_zh"]
    src_lang_to_name = {
        "ar_zh": 'Chinese+Arabic',
        "en_zh": 'English+Chinese',
        "en_ar": 'English+Arabic'
    }
    tgt_langs = ["en", "zh", "ar"]
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            filename = os.path.join(opt.dir, '{}_{}_{}_test.txt'.format(
                src_lang, opt.model_name, tgt_lang))
            if os.path.isfile(filename):
                with open(filename) as f:
                    for line in f:
                        line = line.strip()
                        out = re.findall(pattern, line)
                        if out:
                            assert len(out) == 1 and len(out[0]) == 3
                            result['{}_{}'.format(src_lang, tgt_lang)] = [float(v) for v in out[0]]
                            break

    table = PrettyTable()
    table.field_names = ["", "English", "Chinese", "Arabic", "Average"]
    table.align[""] = "l"
    table.align["English"] = "c"
    table.align["Chinese"] = "c"
    table.align["Arabic"] = "c"
    for src_lang in src_langs:
        f1_scores = []
        scores = []
        for tgt_lang in tgt_langs:
            label = '{}_{}'.format(src_lang, tgt_lang)
            if label in result:
                f1_scores.append(result[label][2])
                score = '/'.join(['{:0.2f}'.format(v) for v in result[label]])
                scores.append(score)
            else:
                scores.append('NA')

        average = '{:0.2f}'.format(sum(f1_scores) / len(f1_scores)) \
            if f1_scores else 'NA'
        table.add_row([src_lang_to_name[src_lang], *scores, average])
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--model_name', type=str, required=True, help="model file name")
    parser.add_argument('--dir', type=str, required=True, help="directory path")
    parser.add_argument('--multi_source', type='bool', default=False)
    args = parser.parse_args()
    if args.multi_source:
        multi_source(args)
    else:
        single_source(args)
