#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp
SEED=1111

declare -A LANG_MAP
LANG_MAP['en']='English'
LANG_MAP['ar']='Arabic'
LANG_MAP['zh']='Chinese'

if [[ ! -d $DATA_DIR ]]; then
    echo "${DATA_DIR} does not exist"
    exit 1
fi

if [[ ! -d $MODEL_DIR ]]; then
    echo "${MODEL_DIR} does not exist, creating the directory"
    mkdir $MODEL_DIR
fi


BERT_FEAT_DIR=${DATA_DIR}/bert_features/ace_relation
#BERT_FEAT_DIR=${DATA_DIR}/xlmroberta_features/ace_relation

OPTIM=sgd
LR=0.1
#OPTIM=adam
#LR=0.001

USE_BERT=True
if [[ $USE_BERT == True ]]; then
    USE_WORD=False
else
    USE_WORD=True
fi


function train () {

echo "============TRAINING============"

eval "LANG=($2)"
MODEL_NAME=$3
TARGET_DIR=$4
make_dir $TARGET_DIR

export PYTHONPATH=$SRC_DIR
export CUDA_VISIBLE_DEVICES=$1

python -W ignore ${SRC_DIR}/main/main.py \
--random_seed $SEED \
--data_workers 5 \
--language ${LANG[*]} \
--use_bert $USE_BERT \
--use_word $USE_WORD \
--data_dir ${DATA_DIR}/ace_relation/ \
--embed_dir ${DATA_DIR}/ace_relation/ \
--embedding_file aligned.embed.300.vec \
--bert_feats $BERT_FEAT_DIR \
--train_filename train \
--valid_filename dev \
--vocab_file vocab.txt \
--max_examples -1 \
--fix_embeddings True \
--batch_size 50 \
--test_batch_size 50 \
--num_epochs 200 \
--pos_dim 30 \
--ner_dim 30 \
--deprel_dim 30 \
--model_type gcn \
--rnn_hid 0 \
--gcn_hid 200 \
--rnn_layers 1 \
--gcn_layers 2 \
--pool_type max \
--use_sent_rep False \
--pooling_l2 0.0 \
--mlp_layers 2 \
--dropout_emb 0.5 \
--dropout_rnn 0.5 \
--dropout_gcn 0.5 \
--early_stop 20 \
--prune_k 1 \
--optimizer $OPTIM \
--learning_rate $LR \
--lr_decay 0.9 \
--warmup_epochs 0 \
--decay_epoch 5 \
--max_grad_norm 5.0 \
--valid_metric f1 \
--checkpoint True \
--model_dir $TARGET_DIR \
--model_name $MODEL_NAME;

}


function test () {

echo "============TESTING============"

eval "LANG=($2)"
MODEL_NAME=$3
TARGET_DIR=$4

if [[ ! -d $TARGET_DIR ]]; then
    echo "${TARGET_DIR} does not exist"
    exit 1
fi

export PYTHONPATH=$SRC_DIR
export CUDA_VISIBLE_DEVICES=$1

python -W ignore ${SRC_DIR}/main/main.py \
--random_seed $SEED \
--only_test True \
--data_workers 5 \
--language ${LANG[*]} \
--data_dir ${DATA_DIR}/ace_relation/ \
--bert_feats $BERT_FEAT_DIR \
--valid_filename test \
--test_batch_size 50 \
--model_dir $TARGET_DIR \
--model_name $MODEL_NAME;

}


function single_source_transfer() {

TARGET_DIR=${MODEL_DIR}/$2_single
LOG_FILENAME=${TARGET_DIR}/full.log

for src_lang in ar en zh; do
    model_name=${src_lang}_$2
    train $1 $src_lang ${model_name} ${TARGET_DIR}
    for tgt_lang in ar en zh; do
        test $1 $tgt_lang ${model_name} ${TARGET_DIR} |& tee -a $LOG_FILENAME
    done
done

python -W ignore ../preparer.py --model_name $2 \
    --dir ${TARGET_DIR} --multi_source False |& tee -a $LOG_FILENAME

}


function multi_source_transfer() {

declare -a src_langs=("en ar" "ar zh" "en zh")
TARGET_DIR=${MODEL_DIR}/$2_multi
LOG_FILENAME=${TARGET_DIR}/full.log

for i in "${!src_langs[@]}"; do
    model_name=${src_langs[$i]// /_}_$2
    train $1 "${src_langs[$i]}" ${model_name} ${TARGET_DIR}
    for tgt_lang in en ar zh; do
        test $1 $tgt_lang ${model_name} ${TARGET_DIR} |& tee -a $LOG_FILENAME
    done
done

python -W ignore ../preparer.py --model_name $2 \
    --dir ${TARGET_DIR} --multi_source True |& tee -a $LOG_FILENAME

}


single_source_transfer  $1 $2
multi_source_transfer $1 $2
