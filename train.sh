# #!/bin/bash
# source ~/.bashrc
# conda init
# conda activate dfine
# export model=n
# CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_n_Ghostv2.yml --use-amp --seed=0


/home/kk/anaconda3/envs/kkMultimodal/bin/python train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
  --use-amp \
  --output-dir output/dfine_rot180_dual_backbone