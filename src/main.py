import os, sys
import argparse

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # /workspace/code/
from manager import DLManager
from utils.config import get_cfg

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='/workspace/code/configs/config.yaml')
parser.add_argument('--data_root', type=str, default='/workspace/data/')
parser.add_argument('--save_root', type=str, default='/workspace/code/save')
parser.add_argument('--checkpoint_path', type=str, default='')

parser.add_argument('--save_term', type=int, default=5) #10

parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--save_term', type=int, default=1)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

# Set Config
cfg = get_cfg(args.config_path)
args.freeze_mode = cfg.FREEZE_MODE

exp_manager = DLManager(args, cfg)
exp_manager.train()
exp_manager.test()
