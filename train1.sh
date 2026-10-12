#!/bin/bash
source ~/.bashrc
conda init
conda activate dfine
export model=n
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml --use-amp --seed=0