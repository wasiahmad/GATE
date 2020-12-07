#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp
SEED=1013

declare -A LANG_MAP
LANG_MAP['en']='English'
LANG_MAP['ar']='Arabic'
LANG_MAP['zh']='Chinese'

if [[ ! -d $DATA_DIR ]]; then
    echo "${DATA_DIR} does not exist"
    exit 1
fi

if [[ ! -d $MODEL_DIR ]]; then
    make_dir $MODEL_DIR
fi


BERT_FEAT_DIR=${DATA_DIR}/bert_features/ace_relation
#BERT_FEAT_DIR=${DATA_DIR}/xlmroberta_features/ace_relation

OPTIM=sgd
LR=0.1
LR_DECAY=0.9
#OPTIM=adam
#LR=0.0001
#LR_DECAY=0.9

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
--data_dir ${DATA_DIR}/ace_relation \
--embed_dir ${DATA_DIR}/ace_relation \
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
--model_type gtn \
--embed_graph 4 \
--max_tree_dist 2 2 4 4 \
--max_src_len 1000 \
--struct_position False \
--position_dim 0 \
--max_relative_pos 0 \
--use_neg_dist True \
--tran_hid 512 \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--gcn_hid 0 \
--tran_layers 1 \
--gcn_layers 2 \
--pool_type max \
--mlp_layers 2 \
--dropout_emb 0.5 \
--dropout_gcn 0.5 \
--trans_drop 0.5 \
--early_stop 20 \
--prune_k 1 \
--optimizer $OPTIM \
--learning_rate $LR \
--lr_decay $LR_DECAY \
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
--data_dir ${DATA_DIR}/ace_relation \
--bert_feats $BERT_FEAT_DIR \
--valid_filename test \
--test_batch_size 50 \
--model_dir $TARGET_DIR \
--model_name $MODEL_NAME;

}


function single_source_transfer() {

# read the split words into an array based on comma delimiter
IFS=',' read -a GPU_IDS <<< "$1"
if [[ ${#GPU_IDS[@]} -eq 1 ]]; then
    echo "Warning: one GPU is not enough to run for Arabic dataset"
fi

TARGET_DIR=${MODEL_DIR}/$2_single
LOG_FILENAME=${TARGET_DIR}/full.log

for src_lang in ar en zh; do
    if [[ $src_lang == "ar" ]]; then gpu_id=$1; else gpu_id=${GPU_IDS[0]}; fi
    train "$gpu_id" $src_lang ${src_lang}_$2 ${TARGET_DIR}
    for tgt_lang in ar en zh; do
        if [[ $tgt_lang == "ar" ]]; then gpu_id=$1; else gpu_id=${GPU_IDS[0]}; fi
        test "$gpu_id" $tgt_lang ${src_lang}_$2 ${TARGET_DIR} |& tee -a $LOG_FILENAME
    done
done

python -W ignore ../preparer.py --model_name $2 \
    --dir ${TARGET_DIR} --multi_source False |& tee -a $LOG_FILENAME

}


function multi_source_transfer() {

# read the split words into an array based on comma delimiter
IFS=',' read -a GPU_IDS <<< "$1"
if [[ ${#GPU_IDS[@]} -eq 1 ]]; then
    echo "Warning: one GPU is not enough to run for Arabic dataset"
fi

declare -a src_langs=("en ar" "ar zh" "en zh")
TARGET_DIR=${MODEL_DIR}/$2_multi
LOG_FILENAME=${TARGET_DIR}/full.log

for i in "${!src_langs[@]}"; do
    if [[ ${src_langs[$i]} == *"ar"* ]]; then gpu_id=$1; else gpu_id=${GPU_IDS[0]}; fi
    model_name=${src_langs[$i]// /_}_$2
    train "$gpu_id" "${src_langs[$i]}" ${model_name} ${TARGET_DIR}
    for tgt_lang in en ar zh; do
        if [[ $tgt_lang == *"ar"* ]]; then gpu_id=$1; else gpu_id=${GPU_IDS[0]}; fi
        test "$gpu_id" $tgt_lang ${model_name} ${TARGET_DIR} |& tee -a $LOG_FILENAME
    done
done

python -W ignore ../preparer.py --model_name $2 \
    --dir ${TARGET_DIR} --multi_source True |& tee -a $LOG_FILENAME

}


single_source_transfer  $1 $2
multi_source_transfer $1 $2
