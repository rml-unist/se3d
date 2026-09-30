# Transfer to DSEC-3DOD (paper Section VI)

These scripts reproduce the controlled transfer experiments: each model is trained
on real DSEC-3DOD data twice with the same target data, schedule and seed, once
from random initialization (scratch) and once from an SE3D checkpoint.

| Model | SE3D checkpoint used for initialization | Target seeds |
|---|---|---|
| DSGN-event | DSGN-event after 8 SE3D epochs | 20260909, 20260910, 20260911 |
| EMOD | EMOD after 8 SE3D epochs | 20260909 |
| SE-CFF (depth only) | stereo network of the same EMOD checkpoint | 20260909 |

The detection prediction layers (class, box, centerness) are not copied; they keep
the seeded initialization, so both arms of a pair start from identical heads.

## Data

Download from the original sources (not redistributed here):

* **DSEC** (https://dsec.ifi.uzh.ch), train split: events, disparity, calibration
  and the image timestamps of the 34 Zurich sequences listed in the protocol.
* **DSEC-3DOD** (https://github.com/mickeykang16/Ev3DOD): `train.txt`, `val.txt`
  and the per-chunk keyframe annotations `<chunk>/<chunk>_fov_bbox_lidar_check.pkl`.

```
<DSEC>/train/<sequence>/events/{left,right}/{events.h5,rectify_map.h5}
                       /disparity/event/<frame>.png
                       /calibration/{cam_to_cam.yaml,cam_to_lidar.yaml}
                       /images/timestamps.txt
<DSEC-3DOD>/train.txt, val.txt
<DSEC-3DOD>/<chunk>/<chunk>_fov_bbox_lidar_check.pkl
```

Check that a local copy matches the one used in the paper (split files,
annotation and calibration hashes, keyframe lists and target anchors):

```bash
python transfer/check_data.py --dsec-root <DSEC>/train --labels-root <DSEC-3DOD>
```

## Split

`protocol/dsec_paired_v2.json` fixes the keyframes; depth and detection use the same ones.

| Split | Source | Chunks | Keyframes |
|---|---|---:|---:|
| train | official train chunks not held out | 126 | 3,906 |
| validation (model selection) | 14 official train chunks, the first 14 by SHA256 of `se3d-dsec-model-selection-v1/<chunk>` | 14 | 434 |
| test | official DSEC-3DOD validation chunks | 38 | 1,178 |

Each keyframe uses the last 5,000,000 raw events before its timestamp in each
camera, rectified and cut to the common left/right time window, stacked into 10
mixed-density stacks. Boxes stay in the DSEC-3DOD LiDAR frame for evaluation and
are converted with the provided extrinsics for training. Target anchor sizes
(`protocol/dsec_training_anchors_v2.json`) are medians of the training boxes.

## Environments

* Training and inference: the PyTorch environment of the repository (Python 3.9,
  PyTorch 2.5.1). DSGN-event target training peaks at about 11 GB of GPU memory.
* Detection metrics: a separate CPU environment with TensorFlow 2.12 and
  `waymo-open-dataset-tf-2-12-0==1.6.2`:
  ```bash
  python3.9 -m venv metrics_env && metrics_env/bin/pip install -r transfer/requirements-metrics.txt
  ```
  Pass its interpreter as `--metrics-python metrics_env/bin/python`.

## Commands

Optional cache of the event stacks (training reads identical tensors with or without it):

```bash
python transfer/prepare_cache.py --dsec-root <DSEC>/train --cache-root <cache>
```

Training (16 epochs, 62,496 updates, batch size 1, Adam with lr 1e-4 and weight
decay 1e-4, no augmentation; best.pth is the epoch with the highest validation
V/P Level-2 AP, or the lowest disparity MAE for SE-CFF):

```bash
COMMON="--dsec-root <DSEC>/train --labels-root <DSEC-3DOD> --metrics-python metrics_env/bin/python"
for SEED in 20260909 20260910 20260911; do
  python transfer/train.py --model dsgn_event --init scratch --seed $SEED --output runs/dsec_dsgn_event_scratch_$SEED $COMMON
  python transfer/train.py --model dsgn_event --init se3d --source weights/se3d_dsgn_event_8ep.pth \
      --seed $SEED --output runs/dsec_dsgn_event_se3d_$SEED $COMMON
done
python transfer/train.py --model emod --init scratch --output runs/dsec_emod_scratch $COMMON
python transfer/train.py --model emod --init se3d --source weights/se3d_emod_8ep.pth --output runs/dsec_emod_se3d $COMMON
python transfer/train.py --model se_cff --init scratch --output runs/dsec_se_cff_scratch $COMMON
python transfer/train.py --model se_cff --init se3d --source weights/se3d_emod_8ep.pth --output runs/dsec_se_cff_se3d $COMMON
```

Test the trained target model on the 1,178 keyframes:

```bash
python transfer/test.py --model dsgn_event --checkpoint runs/dsec_dsgn_event_se3d_20260909/best.pth \
    --output results/dsec_dsgn_event_se3d_20260909 $COMMON
```

`test.py` writes `fixed_test_predictions.pkl` and `test_metrics.json`. The
predictions can be re-scored without a GPU:

```bash
metrics_env/bin/python transfer/evaluate_waymo.py results/.../fixed_test_predictions.pkl metrics.json
```

## Results in the paper (Table VI)

Waymo Level-2 3D AP (%) on the test keyframes; V/P is the mean of Vehicle and
Pedestrian AP; MAE is the disparity error (px) at the keyframes.

| Model | Init | Seed | Vehicle | Pedestrian | V/P | MAE | RMSE | Selected epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DSGN-event | scratch | 20260909 | 4.79 | 0.62 | 2.70 | 0.870 | 1.990 | 15 |
| DSGN-event | SE3D | 20260909 | 7.81 | 3.09 | 5.45 | 0.829 | 1.839 | 8 |
| DSGN-event | scratch | 20260910 | 6.02 | 0.20 | 3.11 | 0.866 | 1.976 | 16 |
| DSGN-event | SE3D | 20260910 | 9.51 | 1.29 | 5.40 | 0.812 | 1.809 | 15 |
| DSGN-event | scratch | 20260911 | 5.34 | 0.56 | 2.95 | 0.887 | 2.050 | 16 |
| DSGN-event | SE3D | 20260911 | 7.45 | 1.64 | 4.55 | 0.846 | 1.909 | 8 |
| DSGN-event | scratch | mean ± SD | 5.38 ± 0.61 | 0.46 ± 0.23 | 2.92 ± 0.20 | 0.874 ± 0.011 | 2.005 | |
| DSGN-event | SE3D | mean ± SD | 8.26 ± 1.10 | 2.01 ± 0.95 | 5.13 ± 0.51 | 0.829 ± 0.017 | 1.852 | |
| EMOD | scratch | 20260909 | 0.38 | 0.09 | 0.24 | 1.039 | 2.589 | 8 |
| EMOD | SE3D | 20260909 | 2.16 | 0.01 | 1.08 | 0.813 | 1.937 | 13 |
| SE-CFF | scratch | 20260909 | – | – | – | 0.858 | 2.051 | 12 |
| SE-CFF | SE3D | 20260909 | – | – | – | 0.743 | 1.774 | 12 |

Supplementary Sec. 10 also trains the DSGN-event SE3D arm (seed 20260909) from a
source trained on the deduplicated SE3D annotations (V/P AP 6.25%, MAE 0.814 px).
For this comparison, train the source model with
`python tools/train.py --model dsgn_event --data-root /data/SE3D --labels corrected --epochs 8 --output runs/se3d_dsgn_event_corrected_8ep`,
then pass `--source runs/se3d_dsgn_event_corrected_8ep/last.pth` to the target
training command above.

Scores obtained on another GPU type differ slightly from these (in our checks,
box coordinates by about 1e-3 m and scores by about 1e-4 on identical inputs).
The single-seed experiment on the interpolated 100-Hz annotations
(Suppl. Sec. 7.3) uses a separate two-GPU runner and is not included here.

## Third-party code

* `se_cff/components/models`: SE-CFF (Nam et al., CVPR 2022), MIT license per
  its repository; the deformable convolution uses `torchvision.ops.deform_conv2d`.
* `vendor/waymo_eval_detection.py`: unmodified Waymo evaluation wrapper from
  Ev-3DOD (MIT), which follows OpenPCDet's wrapper (Apache-2.0); it calls the
  metric operators of the `waymo-open-dataset` package.
