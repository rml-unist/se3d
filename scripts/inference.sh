#!/bin/bash

set -x

cuda_idx='0'
config_path=/workspace/code/configs/config.yaml
data_root=/workspace/data
save_root=/workspace/data/output/save/sbn_noopt_eo_75
num_workers=16
checkpoint_path=/workspace/data/output/save/sbn_noopt_eo/weights/75.pth

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 ../src/inference.py --data_root ${data_root} --save_root ${save_root} --checkpoint_path ${checkpoint_path} --config_path ${config_path} --num_workers ${num_workers}

