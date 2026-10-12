#!/bin/bash
source ~/.bashrc
conda init
conda activate dfine
export model=n
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_n_Ghostv2.yml --test-only -r output/dfine_hgnetv2_n_coco_DFINE_origin/best_stg1.pth