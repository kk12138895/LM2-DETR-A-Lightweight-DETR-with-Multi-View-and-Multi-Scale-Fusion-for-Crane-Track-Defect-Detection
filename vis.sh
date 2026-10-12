#!/bin/bash
source ~/.bashrc
conda init
conda activate dfine
export model=n
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 -m ./tools/visualization/fiftyone_vis.py -c configs/dfine/custom/dfine_hgnetv2_n_Ghostv2.yml -r output/dfine_hgnetv2_n_coco_DFINE_RSDDsv2/best_stg1.pth