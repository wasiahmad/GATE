#!/usr/bin/env bash

SRC_DIR=..

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

function bert_features () {

export CUDA_VISIBLE_DEVICES=$1
BERT_MODEL=$2
DATA_DIR=$3
TGT_DIR=$4

if [[ ! -d $TGT_DIR ]]; then
    for lang in ar zh;
    do
        echo "Extracting features for lang = $lang"
        python -W ignore ${SRC_DIR}/bert_feats.py \
        --src_dir $DATA_DIR \
        --tgt_dir $TGT_DIR \
        --bert_model $BERT_MODEL \
        --lang $lang
    done
fi

}

URL_PREFIX='https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131'
OUT_DIR=./udpipe/
if [[ ! -d $OUT_DIR ]]; then
    echo "Downloading UDPipe models"
    mkdir $OUT_DIR
    FILENAME=english-ewt-ud-2.5-191206.udpipe
    curl -o ${OUT_DIR}/${FILENAME} ${URL_PREFIX}/${FILENAME}
    FILENAME=arabic-padt-ud-2.5-191206.udpipe
    curl -o ${OUT_DIR}/${FILENAME} ${URL_PREFIX}/${FILENAME}
    FILENAME=chinese-gsd-ud-2.5-191206.udpipe
    curl -o ${OUT_DIR}/${FILENAME} ${URL_PREFIX}/${FILENAME}
fi

make_dir ace_event
make_dir ace_event/Arabic
make_dir ace_event/Chinese
make_dir ace_relation
make_dir ace_relation/Arabic
make_dir ace_relation/Chinese

python collect_sentence.py
python translate.py
python preprocess.py

bert_features $1 "bert-base-multilingual-cased" "ace_event" "bert_features/ace_event"
bert_features $1 "bert-base-multilingual-cased" "ace_relation" "bert_features/ace_relation"

bert_features $1 "xlm-roberta-base" "ace_event" "xlmroberta_features/ace_event"
bert_features $1 "xlm-roberta-base" "ace_relation" "xlmroberta_features/ace_relation"
