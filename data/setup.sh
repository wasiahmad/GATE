#!/usr/bin/env bash

SRC_DIR=..

############################# Downloading Embeddings #############################

URL_PREFIX='https://dl.fbaipublicfiles.com/fasttext/vectors-aligned'
OUT_DIR=./fasttext_aligned
if [[ ! -d $OUT_DIR ]]; then
    echo "Downloading fasttext (aligned) vectors"
    mkdir $OUT_DIR
    for lang in en ar zh;
    do
        FILENAME=wiki.${lang}.align.vec
        curl -o ${OUT_DIR}/${FILENAME} ${URL_PREFIX}/${FILENAME}
    done
fi

URL_PREFIX='https://dl.fbaipublicfiles.com/fasttext/vectors-crawl'
OUT_DIR=./fasttext_multilingual
if [[ ! -d $OUT_DIR ]]; then
    echo "Downloading fasttext (multilingual) embeddings"
    mkdir $OUT_DIR
    for lang in en ar zh;
    do
        FILENAME=cc.${lang}.300.vec.gz
        curl -o ${OUT_DIR}/${FILENAME} ${URL_PREFIX}/${FILENAME}
        gzip -d ${OUT_DIR}/${FILENAME}
    done
fi

############################# Downloading ACE dataset #############################

data_download=false

# Due to data privacy we cannot share the data

############################# Aggregating Statistics #############################

if [[ ! $data_download ]]; then
    echo "Collecting statistics for ACE datasets"
    PYTHONPATH=$SRC_DIR python -W ignore get_stat.py
fi

############################## Filtering Embeddings ##############################

TGT_DIR=ace_event
TGT_FILE=aligned.embed.300.vec
if [[ ! -f ${TGT_DIR}/${TGT_FILE} ]]; then
    echo "Extracting aligned embeddings for ACE event tokens"
    python -W ignore filter_emb.py \
        --src_dir ./fasttext_aligned \
        --tgt_dir ${TGT_DIR} \
        --src_file wiki.en.align.vec wiki.ar.align.vec wiki.zh.align.vec \
        --tgt_file ${TGT_FILE} \
        --lang en ar zh \
        --vocab_file ${TGT_DIR}/vocab.txt
fi

TGT_DIR=ace_relation
TGT_FILE=aligned.embed.300.vec
if [[ ! -f ${TGT_DIR}/${TGT_FILE} ]]; then
    echo "Extracting aligned embeddings for ACE relation tokens"
    python -W ignore filter_emb.py \
        --src_dir ./fasttext_aligned \
        --tgt_dir ${TGT_DIR} \
        --src_file wiki.en.align.vec wiki.ar.align.vec wiki.zh.align.vec \
        --tgt_file ${TGT_FILE} \
        --lang en ar zh \
        --vocab_file ${TGT_DIR}/vocab.txt
fi

###################### Extracting BERT/XLM-RoBERTa Features ######################

function bert_features () {

export CUDA_VISIBLE_DEVICES=$1
BERT_MODEL=$2
DATA_DIR=$3
TGT_DIR=$4

if [[ ! -d $TGT_DIR ]]; then
    for lang in en ar zh;
    do
        echo "Extracting features for lang = $lang"
        PYTHONPATH=$SRC_DIR python -W ignore bert_feats.py \
        --src_dir $DATA_DIR \
        --tgt_dir $TGT_DIR \
        --bert_model $BERT_MODEL \
        --lang $lang
    done
fi

}

bert_features $1 "bert-base-multilingual-cased" "ace_event" "bert_features/ace_event"
bert_features $1 "bert-base-multilingual-cased" "ace_relation" "bert_features/ace_relation"

bert_features $1 "xlm-roberta-base" "ace_event" "xlmroberta_features/ace_event"
bert_features $1 "xlm-roberta-base" "ace_relation" "xlmroberta_features/ace_relation"
