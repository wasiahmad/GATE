#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

EMBED_DIR=../fasttext_aligned

URL_PREFIX='https://dl.fbaipublicfiles.com/fasttext/vectors-aligned'
if [[ ! -d $EMBED_DIR ]]; then
    echo "Downloading fasttext (aligned) vectors"
    mkdir $EMBED_DIR
    for lang in en ar zh;
    do
        FILENAME=wiki.${lang}.align.vec
        curl -o ${EMBED_DIR}/${FILENAME} ${URL_PREFIX}/${FILENAME}
    done
fi

function knn () {

src_dir=$1
tgt_dir=$2
src_lang=$3
tgt_lang=$4

python -W ignore find_knn.py \
--src_vocab ${src_dir}/vocab.txt \
--src_emb ${EMBED_DIR}/wiki.${src_lang}.align.vec \
--tgt_emb ${EMBED_DIR}/wiki.${tgt_lang}.align.vec \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--center \
--maxload 200000 \
--knn 3 \
--tgt_file ${tgt_dir}/${src_lang}_${tgt_lang}_knn.txt

}

function run_knn () {

SRC_DIR=$1
TGT_DIR=$2
make_dir $TGT_DIR
for src_lang in en ar zh; do
    for tgt_lang in en ar zh; do
        if [[ $src_lang != $tgt_lang ]]; then
            echo "[Language: source-${src_lang} target-${tgt_lang}]"
            knn $SRC_DIR $TGT_DIR $src_lang $tgt_lang
        fi
    done
done

}

run_knn ../ace_event ./ace_event
run_knn ../ace_relation ./ace_relation

function process_vocab_embeds () {

SRC_DIR=$1
TGT_DIR=$2
TGT_FILE=aligned.embed.300.vec
if [[ ! -f ${TGT_DIR}/${TGT_FILE} ]]; then
    python -W ignore process.py \
        --src_dir $EMBED_DIR \
        --tgt_dir $TGT_DIR \
        --src_file wiki.en.align.vec wiki.ar.align.vec wiki.zh.align.vec \
        --tgt_file $TGT_FILE \
        --lang en ar zh \
        --vocab_file ${SRC_DIR}/vocab.txt
fi

}

process_vocab_embeds ../ace_event ./ace_event
process_vocab_embeds ../ace_relation ./ace_relation
