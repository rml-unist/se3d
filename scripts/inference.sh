#!/bin/bash

set -x

cuda_idx='0'
config_path=/workspace/code/configs/config.yaml
# config_path=/csha/se-cff/configs/config_ori.yaml
# data_root=/workspace/data/CARLA/training_slice
# data_root=/csha/DSEC
# data_root=/workspace/data/map6
data_root=/workspace/data
# data_root=/workspace/data
# save root 바꿔줘야함!!!!
# save_root=/workspace/data/output/save/test
save_root=/workspace/data/output/save/sbn_optic_epi_newdist
num_workers=16
# checkpoint_path=/csha/csod/save/trial_6/weights/100.pth
# checkpoint_path=/csha/csod/save/trial_6/weights/final.pth
# checkpoint_path=/csha/csod/final.pth
checkpoint_path=/workspace/data/output/save/sbn_optic_epi_newdist/weights/25.pth
#checkpoint_path=/csha/csod/60.pth

# CUDA_VISIBLE_DEVICES=${cuda_idx} python3 ../src/inference.py --data_root ${data_root} --save_root ${save_root} --checkpoint_path ${checkpoint_path}
CUDA_VISIBLE_DEVICES=${cuda_idx} python3 ../src/inference.py --data_root ${data_root} --save_root ${save_root} --checkpoint_path ${checkpoint_path} --config_path ${config_path} --num_workers ${num_workers}

