#!/bin/bash

set -x

cuda_idx='0,1'
config_path=/workspace/code/configs/config.yaml
data_root=/workspace/data
save_root=/workspace/data/output/save/sbn_noopt_eo
checkpoint_path=/workspace/data/output/save/sbn_noopt_eo/weights/final.pth

num_workers=8
NUM_PROC=2

# Check if checkpoint_path is set and not commented out
if [ -n "$checkpoint_path" ]; then
  checkpoint_arg="--checkpoint_path ${checkpoint_path}"
else
  checkpoint_arg=""
fi

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py \
 --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} ${checkpoint_arg}
