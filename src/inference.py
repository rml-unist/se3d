import os
import argparse

import torch
import sys
# 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manager import DLManager
from utils.config import get_cfg



torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='/workspace/code/configs/config.yaml')
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_root', type=str, required=True)

parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

assert os.path.isdir(args.data_root)

# Set Config
print(args.config_path)
cfg = get_cfg(args.config_path)
args.freeze_mode = cfg.FREEZE_MODE

exp_manager = DLManager(args, cfg)
exp_manager.load(args.checkpoint_path)

exp_manager.test()
