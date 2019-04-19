#!/usr/bin/env bash

python demo.py \
    --model-name 'reid-PCB' \
    --data-dir './datasets/Market-1501/market-reid' \
    --query-index 2 \
    --num-images 10
