#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip install --upgrade google-cloud-translate

import os
import json
import time
from tqdm import tqdm
from google.cloud import translate_v2 as translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_cloud_credentials.json"


# src: https://cloud.google.com/translate/docs/basic/translating-text#translating_text
def translate_text(text, source=None, target='en'):
    translate_client = translate.Client()
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text,
        source_language=source,
        target_language=target
    )

    return [r['translatedText'] for r in result]


def get_parallel_data(srcfile, tgtfile, source, target):
    with open(srcfile) as f:
        data = json.load(f)

    input_sentences = []
    for ex in tqdm(data, total=len(data)):
        input_sentences.append(ex['sentence'])

    translated_sentences = []
    batch_size = 32
    print('Total batches - ', len(input_sentences) // batch_size)
    with open("temp.txt", "w") as outfile:
        for idx, sent in enumerate(tqdm(input_sentences, total=len(input_sentences))):
            t_sentences = translate_text([sent], source, target)
            translated_sentences.extend(t_sentences)
            outfile.write(t_sentences[0] + '\n')
            outfile.flush()
            if (idx + 1) % batch_size == 0:
                time.sleep(60)

    parallel_data = []
    for idx, ex in enumerate(data):
        parallel_data.append({
            'id': ex['id'],
            'token': ex['token'],
            'sentence': ex['sentence'],
            'parallel': translated_sentences[idx],
            'subj': ex['subj'],
            'obj': ex['obj'],
            'subj_type': ex['subj_type'],
            'obj_type': ex['obj_type'],
            'relation': ex['relation']
        })

    with open(tgtfile, 'w', encoding='utf-8') as fw:
        json.dump(parallel_data, fw, sort_keys=True, indent=4)


if __name__ == '__main__':
    get_parallel_data('event_en_test.json',
                      'event_en_test_zh.json',
                      source='en',
                      target='zh')
    get_parallel_data('relation_en_test.json',
                      'relation_en_test_zh.json',
                      source='en',
                      target='zh')
    get_parallel_data('ace_event/en_test.json',
                      'ace_event/en_test_ar.json',
                      source='en',
                      target='ar')
    get_parallel_data('ace_relation/en_test.json',
                      'ace_relation/en_test_ar.json',
                      source='en',
                      target='ar')
