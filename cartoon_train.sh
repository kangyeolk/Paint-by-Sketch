#!/bin/bash

# bash cartoon_train.sh 2,3

# CUDA_VISIBLE_DEVICES=<gpu_ids> python -u main.py \
# --logdir <path/to/save_artifacts> \
# --base <path/to/config> \
# --pretrained_model pretrained_models/model-modified-12channel.ckpt \
# --scale_lr False

CUDA_VISIBLE_DEVICES=$1 python -u main.py \
--logdir $2 \
--base $3 \
--pretrained_model pretrained_models/model-modified-12channel.ckpt \
--scale_lr False