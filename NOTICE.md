# Third-party code

| Location | Origin | License |
|---|---|---|
| `dsgn_event/` | [DSGN](https://github.com/JIA-Lab-research/DSGN) (Chen et al., CVPR 2020), commit in `dsgn_event/UPSTREAM.json`, adapted to 10-channel event input; the cost-volume operator is reimplemented in PyTorch | MIT (`dsgn_event/LICENSE`) |
| `emod/src/lib/dsgn/` | DSGN detection head, loss and box utilities | MIT |
| `emod/src/lib/dsgn/layers/`, `dsgn_event/dsgn/layers/` | utilities derived from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark); unused local extension build sources have been removed | MIT |
| `emod/src/lib/se_cff/`, `transfer/se_cff/` | [SE-CFF](https://github.com/yonseivnl/se-cff) (Nam et al., CVPR 2022) | MIT, as stated in its README |
| `emod/src/lib/datasets/dsec/event/sbn/slice.py` | event slicing from the [DSEC](https://github.com/uzh-rpg/DSEC) tools | MIT |
| `emod/src/utils/evalod.py`, `kitti_common.py`, `rotate_iou.py`, `emod/src/lib/dsgn/eval/` | KITTI evaluation from [second.pytorch](https://github.com/traveller59/second.pytorch), modified for the SE3D classes and difficulty levels | MIT |
| `transfer/vendor/waymo_eval_detection.py` | Waymo evaluation wrapper from [Ev-3DOD](https://github.com/mickeykang16/Ev3DOD), which follows [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | MIT; OpenPCDet is Apache-2.0 |

SE3D was generated with [CARLA](https://github.com/carla-simulator/carla) 0.9.15
and a modified [CARLA-KITTI](https://github.com/fnozarian/CARLA-KITTI) collector;
neither is included here.

The SE3D dataset's MIT terms are stated separately in
[`docs/DATASET_LICENSE.md`](docs/DATASET_LICENSE.md). CARLA-specific assets
retain their upstream CC-BY terms; see the versioned sources and attribution in
[`docs/DATASET_NOTICE.md`](docs/DATASET_NOTICE.md). The dataset MIT grant covers
the SE3D authors' rights and does not relicense third-party assets.
