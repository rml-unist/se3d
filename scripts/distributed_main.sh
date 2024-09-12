#!/bin/bash

set -x

# cuda_idx='0,1,2,3,4,5,6'
# cuda_idx='1,2'
cuda_idx='0,1,2,3,4'
# config_path=/csha/csod1/configs/config.yaml
# config_path=/csha/se-cff/configs/config_ori.yaml
config_path=/workspace/code/configs/config.yaml
# data_root=/workspace/data/map6
data_root=/workspace/data
# data_root=/workspace/data
# data_root=/csha/DSEC
# save_root=/workspace/data/output/save/trial_11
save_root=/workspace/data/output/save/sbn_noopt_eo_jjinmak
checkpoint_path=/workspace/data/output/save/sbn_noopt_eo_jjinmak/weights/final.pth

num_workers=20
NUM_PROC=5

# Check if checkpoint_path is set and not commented out
if [ -n "$checkpoint_path" ]; then
  checkpoint_arg="--checkpoint_path ${checkpoint_path}"
else
  checkpoint_arg=""
fi

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py \
 --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} ${checkpoint_arg}
