#!/bin/bash

set -x

RANK=0
WORLD_SIZE=1

python pretrain_bert.py \
    --batch-size 4 \
    --tokenizer-type BytePairTokenizer \
    --cache-dir cache_dir \
    --tokenizer-model-type bert-large-uncased \
    --vocab-size 50257 \
    --train-data wikipedia \
    --valid-data wikipedia \
    --test-data wikipedia \
    --loose-json \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 128 \
    --max-position-embeddings 512 \
    --num-layers 24 \
    --hidden-size 1024 \
    --intermediate-size 4096 \
    --num-attention-heads 16 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --train-iters 1000000 \
    --epochs 10 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --lr-decay-iters 990000 \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --fp32-layernorm \
    --fp32-embedding \
    --hysteresis 2 \
    --num-workers 2
