# Transfer to DSEC-3DOD (paper Section VI)

These scripts run the controlled transfer experiments: each model is trained
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

Full-pool training uses 62,496 updates, equivalent to 16 passes through the 3,906
training frames, with validation every 3,906 updates. Batch size is 1, Adam uses
lr 1e-4 and weight decay 1e-4, and there is no augmentation. `best.pth` keeps the
highest validation V/P Level-2 AP (lowest disparity MAE for SE-CFF); exact ties
keep the earliest update. `final.pth` contains the final update independently of
model selection.

Source and target training share the default `--backend-profile historical`.
Use the same explicit profile for both initialization arms and every seed;
source weights do not select the target's numerical settings. The
[backend table](../docs/EXPERIMENTS.md#backend-settings-and-determinism) lists
the optional reproducible profile and strict error checking.

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

## Corrected-source seed and label-efficiency experiments

The corrected-source experiment crosses three source seeds
(`20260909`, `20260910`, `20260911`) with those same three target optimization
seeds. Each source is the corrected DSGN-event checkpoint at update
214,368, which is epoch 8 of the full SE3D training pool. Three scratch runs,
one per target seed, provide shared references. The nine pretrained runs share
source weights and scratch references. Summarize source-seed and target-seed
variation separately, holding the other seed fixed.

Label-efficiency runs use the single fixed corrected source seed `20260909` at
update 214,368. The same nested subsets are used across the three target seeds
and both initialization arms. Whole training chunks are ordered by
`SHA256("se3d-target-label-efficiency-chunks-v1/20261010/" + chunk)`; each subset
takes the first `ceil(fraction * 126)` chunks in that order. Internal validation
and test remain the same 434 and 1,178 keyframes. Subset membership depends
only on chunk names and the fixed subset seed.

| Nominal labels | Actual labels | Chunks | Frames | Empty frames | Vehicle boxes | Pedestrian boxes | Cyclist boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% | 10.3175% | 13 | 403 | 135 | 655 | 4 | 41 |
| 25% | 25.3968% | 32 | 992 | 261 | 1,848 | 230 | 59 |
| 50% | 50.0000% | 63 | 1,953 | 410 | 3,828 | 535 | 98 |

All selected frames, including empty frames, are retained. The 10% subset has
only four Pedestrian boxes. Standard deviations across target seeds describe
optimization variation with subset membership fixed.

Both box and disparity supervision use only the selected training frames.
Anchor dimensions and center heights are recomputed from those frames alone,
and the scratch/pretrained pair shares the resulting anchor file.
If a class has no raw boxes, the constants in
[`absent_class_anchors_v1.json`](protocol/absent_class_anchors_v1.json) are used
only for that class. No fallback is needed for the three subsets above.
Training recomputes these statistics and rejects anchors that differ from
the selected training data.

```bash
python transfer/prepare_subsets.py --dsec-root /data/DSEC/train \
    --labels-root /data/DSEC-3DOD --output runs/target_subsets

SOURCE=/path/to/corrected_dsgn_source_s20260909/step_0214368.pth
SOURCE_SHA256=$(sha256sum "$SOURCE" | cut -d ' ' -f 1)
SUBSETS=runs/target_subsets
DATA=(--dsec-root /data/DSEC/train --labels-root /data/DSEC-3DOD \
      --metrics-python metrics_env/bin/python)
for FRACTION in 10 25 50; do
  for SEED in 20260909 20260910 20260911; do
    PAIR=(--model dsgn_event --backend-profile historical --seed "$SEED" \
          --updates 62496 --validate-every 3906 \
          --subset "$SUBSETS/fraction_$FRACTION/subset.json" \
          --anchors "$SUBSETS/fraction_$FRACTION/anchors.json")
    python transfer/train.py "${DATA[@]}" "${PAIR[@]}" --init scratch \
        --output "runs/labels_${FRACTION}_scratch_$SEED"
    python transfer/train.py "${DATA[@]}" "${PAIR[@]}" --init se3d \
        --source "$SOURCE" --source-sha256 "$SOURCE_SHA256" \
        --source-seed 20260909 --source-step 214368 \
        --output "runs/labels_${FRACTION}_pretrained_$SEED"
  done
done
```

Use `--manifests-only` to write subset lists without reading annotations.
The 100% reference uses the full-pool command without
`--subset` or `--anchors`. For the source/target seed matrix, change the source
checkpoint and `--source-seed` together and retain the same target budget.

Every fraction trains for **62,496 updates**, with **16 validations** at
multiples of 3,906. Smaller subsets repeat more often under this equal update
budget. Use `--updates` and `--validate-every` for subset comparisons;
`--epochs` is a shorthand based on the full training pool.

## Resuming training

Repeat the same command and output directory to resume from `last.pth`.
It restores the model, optimizer, random-number generators, sample order and
validation progress. Interrupted validation finishes before the next update.
Changed code, inputs or settings require a new output directory.

`SIGUSR1`, `SIGTERM` or the time limit saves a checkpoint and returns **75**.
For a 10h30 job, `--max-seconds 37800 --save-margin-seconds 900` requests a save
15 minutes before the end. Use the actual job time limit; `SLURM_JOB_END_TIME`
is also accepted as a Unix timestamp. Resubmit with the same command after
exit 75. `--allocation-updates N` stops after N additional updates for resume
checks.

`effective_config.json` records training settings and input/code hashes;
`initialization.json` records copied and reset weights. `training_complete.json`
contains the completed budget and selected checkpoint. Event HDF5 and cache
contents are not included in the per-run input hashes.

`test.py` uses the checkpoint's anchors and backend settings, and checks the
runtime and protocol. Older checkpoints require their matching protocol and
anchor files. Missing backend flags produce a warning and use the historical
profile. Older `last.pth` files require their original training runner.

## Tests

The CPU tests require the repository's PyTorch environment and no GPU or Waymo
installation:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s transfer/tests -v
```

These check subset selection, anchors, source initialization, backend settings
and interrupted training/validation with small CPU models.

## Historical results in the paper (Table VI)

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
`python tools/train.py --model dsgn_event --data-root /data/SE3D --labels label --updates 214368 --keep-updates 214368 --output runs/se3d_dsgn_event_label_8ep`,
then pass `--source runs/se3d_dsgn_event_label_8ep/step_0214368.pth` to a new target
run. The [historical reproduction limits](../docs/EXPERIMENTS.md#historical-paper-protocol-tables-iv-and-v)
also apply to these results. The supplementary experiment on interpolated
100-Hz annotations (Sec. 7.3) is not included here.

## Third-party code

* `se_cff/components/models`: SE-CFF (Nam et al., CVPR 2022), MIT license per
  its repository; the deformable convolution uses `torchvision.ops.deform_conv2d`.
* `vendor/waymo_eval_detection.py`: unmodified Waymo evaluation wrapper from
  Ev-3DOD (MIT), which follows OpenPCDet's wrapper (Apache-2.0); it calls the
  metric operators of the `waymo-open-dataset` package.
