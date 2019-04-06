#!/bin/bash

set -ex

RANK=0
WORLD_SIZE=1
DATA=~/data/tiny.json

if ! [ -d $(dirname $DATA) ]; then
    mkdir $(dirname $DATA)
fi

if ! [ -f $DATA ]; then
    wget --no-check-certificate https://s3.amazonaws.com/yaroslavvb2/data/tiny.json -O $DATA
fi

echo 'data from $DATA'
python pretrain_bert.py \
    --batch-size 1 \
    --tokenizer-type BytePairTokenizer \
    --cache-dir temp_cache_dir \
    --tokenizer-model-type bert-large-uncased \
    --vocab-size 50257 \
    --train-data $DATA \
    --test-data $DATA \
    --valid-data $DATA \
    --loose-json \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 32 \
    --max-position-embeddings 512 \
    --num-layers 6 \
    --hidden-size 32 \
    --intermediate-size 32 \
    --num-attention-heads 4 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --train-iters 3 \
    --lr 0.00000001 \
    --lr-decay-style linear \
    --lr-decay-iters 990000 \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp32-layernorm \
    --fp32-embedding \
    --hysteresis 2 \
    --num-workers 2 