#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp
SEED=9174

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

function train () {

echo "============TRAINING============"

eval "LANG=($2)"
TGT_LANG=$3
MODEL_NAME=$4
TARGET_DIR=$5
make_dir $TARGET_DIR

export PYTHONPATH=$SRC_DIR
export CUDA_VISIBLE_DEVICES=$1

python -W ignore ${SRC_DIR}/liuetal2019/main.py \
--random_seed $SEED \
--data_workers 5 \
--language ${LANG[*]} \
--tgt_language $TGT_LANG \
--use_word True \
--data_dir ${DATA_DIR}/ace_event/ \
--embed_dir ${DATA_DIR}/cltrans/ace_event/ \
--embedding_file aligned.embed.300.vec \
--vocab_file vocab.txt \
--max_examples -1 \
--fix_embeddings True \
--batch_size 50 \
--test_batch_size 50 \
--num_epochs 200 \
--pos_dim 30 \
--ner_dim 30 \
--deprel_dim 30 \
--type_dim 30 \
--max_src_len 1000 \
--position_dim 0 \
--max_relative_pos 10 \
--use_neg_dist True \
--tran_hid 512 \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--gcn_hid 200 \
--tran_layers 1 \
--gcn_layers 2 \
--pool_type max \
--mlp_layers 2 \
--dropout_emb 0.5 \
--dropout_gcn 0.5 \
--trans_drop 0.2 \
--early_stop 20 \
--prune_k 1 \
--optimizer adam \
--learning_rate 0.001 \
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

python -W ignore ${SRC_DIR}/liuetal2019/main.py \
--random_seed $SEED \
--only_test True \
--data_workers 5 \
--language ${LANG[*]} \
--data_dir ${DATA_DIR}/ace_event/ \
--test_batch_size 50 \
--model_dir $TARGET_DIR \
--model_name $MODEL_NAME;

}

function single_source_transfer() {

TARGET_DIR=${MODEL_DIR}/$2_single
LOG_FILENAME=${TARGET_DIR}/full.log

declare -a src_langs=("en" "en" "ar" "ar" "zh" "zh")
declare -a tgt_langs=("ar" "zh" "en" "zh" "en" "ar")

for i in "${!src_langs[@]}"; do
    src_lang=${src_langs[$i]}
    tgt_lang=${tgt_langs[$i]}
    model_name=${src_lang}_$2_${tgt_lang}
    train $1 $src_lang $tgt_lang $model_name $TARGET_DIR
    test $1 $tgt_lang $model_name $TARGET_DIR |& tee -a $LOG_FILENAME
done

python -W ignore ../preparer.py --model_name $2 \
    --dir ${TARGET_DIR} --multi_source False |& tee -a $LOG_FILENAME

}

function multi_source_transfer() {

TARGET_DIR=${MODEL_DIR}/$2_multi
LOG_FILENAME=${TARGET_DIR}/full.log

declare -a src_langs=("en ar" "ar zh" "en zh")
declare -a tgt_langs=("zh" "en" "ar")

for i in "${!src_langs[@]}"; do
    tgt_lang=${tgt_langs[$i]}
    model_name=${src_langs[$i]// /_}_$2_${tgt_lang}
    train $1 "${src_langs[$i]}" $tgt_lang $model_name $TARGET_DIR
    test $1 $tgt_lang $model_name $TARGET_DIR |& tee -a $LOG_FILENAME
done

python -W ignore ../preparer.py --model_name $2 \
    --dir ${TARGET_DIR} --multi_source True |& tee -a $LOG_FILENAME

}

single_source_transfer  $1 $2
multi_source_transfer $1 $2
