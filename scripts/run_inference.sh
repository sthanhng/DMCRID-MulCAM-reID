#!/usr/bin/env bash

python inference.py \
    --model-path './models' \
    --epoch 'last' \
    --data-dir './datasets/Market-1501/market-reid' \
    --model-name 'reid-ResNet50' \
    --batch-size 32 \
    --num-workers 8