# Third-party code

| Location | Origin | License |
|---|---|---|
| `dsgn_event/` | [DSGN](https://github.com/JIA-Lab-research/DSGN) (Chen et al., CVPR 2020), commit in `dsgn_event/UPSTREAM.json`, adapted to 10-channel event input; the cost-volume operator is reimplemented in PyTorch | MIT (`dsgn_event/LICENSE`) |
| `emod/src/lib/dsgn/` | DSGN detection head, loss and box utilities | MIT |
| `emod/src/lib/dsgn/csrc/`, `dsgn_event/dsgn/csrc/`, `*/layers/` | [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) operators (not compiled by this code) | MIT |
| `emod/src/lib/se_cff/`, `transfer/se_cff/` | [SE-CFF](https://github.com/yonseivnl/se-cff) (Nam et al., CVPR 2022) | MIT, as stated in its README |
| `emod/src/lib/datasets/dsec/event/sbn/slice.py` | event slicing from the [DSEC](https://github.com/uzh-rpg/DSEC) tools | MIT |
| `emod/src/utils/evalod.py`, `kitti_common.py`, `rotate_iou.py`, `emod/src/lib/dsgn/eval/` | KITTI evaluation from [second.pytorch](https://github.com/traveller59/second.pytorch), modified for the SE3D classes and difficulty levels | MIT |
| `transfer/vendor/waymo_eval_detection.py` | Waymo evaluation wrapper from [Ev-3DOD](https://github.com/mickeykang16/Ev3DOD), which follows [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | MIT; OpenPCDet is Apache-2.0 |

SE3D was generated with [CARLA](https://github.com/carla-simulator/carla) 0.9.15
and a modified [CARLA-KITTI](https://github.com/fnozarian/CARLA-KITTI) collector;
neither is included here.
