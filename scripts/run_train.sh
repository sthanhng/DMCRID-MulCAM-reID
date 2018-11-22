#!/usr/bin/env bash

python train.py \
    --model-path 'models' \
    --model-name 'reid-ResNet50' \
    --data-dir './datasets/Market-1501/market-reid' \
    --batch-size 32 \
    --num-epochs 60 \
    --num-workers 8 \
    --weight-decay 5e-4 \
    --lr 0.01 \
    --lr-decay-epoch 30
