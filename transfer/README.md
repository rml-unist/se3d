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

The deferred corrected-source experiment crosses three source seeds
(`20260909`, `20260910`, `20260911`) with those same three target optimization
seeds. Each source is the immutable corrected DSGN-event checkpoint at update
214,368, which is epoch 8 of the full SE3D training pool. Three scratch runs,
one per target seed, provide shared references. Record all nine pretrained
cells; shared source weights and scratch references make these cells correlated.
Report variation across target seeds conditional on one source seed, and across
source seeds conditional on one target seed. Do not interpret the nine cells as
nine independent paired repetitions.

Label-efficiency runs use the single fixed corrected source seed `20260909` at
update 214,368. The same nested subsets are used across the three target seeds
and both initialization arms. Whole training chunks are ordered by
`SHA256("se3d-target-label-efficiency-chunks-v1/20261010/" + chunk)`; each subset
takes the first `ceil(fraction * 126)` chunks in that order. Internal validation
and test remain the same 434 and 1,178 keyframes. The subset seed and identities
were fixed before inspecting their annotations.

| Nominal labels | Actual labels | Chunks | Frames | Empty frames | Vehicle boxes | Pedestrian boxes | Cyclist boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10% | 10.3175% | 13 | 403 | 135 | 655 | 4 | 41 |
| 25% | 25.3968% | 32 | 992 | 261 | 1,848 | 230 | 59 |
| 50% | 50.0000% | 63 | 1,953 | 410 | 3,828 | 535 | 98 |

Counts describe raw selected training annotations; no empty or difficult frames
are removed. The small pedestrian count at 10% does not trigger resampling.
Standard deviations across target seeds measure optimization variation
conditional on this fixed subset, not uncertainty from sampling other subsets.

Both box and disparity supervision use only the selected training frames.
Anchor dimensions and center heights are recomputed from those frames alone,
and the scratch/pretrained pair shares the resulting anchor file. Preparation
and training never open excluded training annotations to obtain these statistics.
If a class has no raw boxes, the predeclared, source-independent constants in
[`absent_class_anchors_v1.json`](protocol/absent_class_anchors_v1.json) are used
only for that class. No fallback is needed for the three frozen subsets above.
Each anchor record contains the selected annotation/calibration hashes, class
counts, and fallback hash. Training recomputes these selected-only statistics
and rejects a mismatched or full-pool anchor file.

```bash
python transfer/prepare_subsets.py --dsec-root /data/DSEC/train \
    --labels-root /data/DSEC-3DOD --output runs/target_subsets

# SOURCE_SHA256 is the SHA256 of the fixed source checkpoint file.
SOURCE=/path/to/corrected_dsgn_source_s20260909/step_0214368.pth
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

Use `--manifests-only` during subset preparation to freeze identities without
reading annotations. The 100% reference uses the full-pool command without
`--subset` or `--anchors`. For the source/target seed matrix, change the source
checkpoint and `--source-seed` together and retain the same target budget.

Every fraction trains for exactly **62,496 optimizer updates** and has the same
**16 validation candidates** at multiples of 3,906. Smaller subsets cycle
deterministically reshuffled permutations until the limit; validation can occur
inside a data pass. This measures equal update budgets, with more passes over
smaller subsets. A nonmultiple update limit used for a short check also receives
a final validation. `--epochs` remains a legacy shorthand based on the full
training pool; explicit `--updates` and `--validate-every` are preferred.

## Resume and provenance

`effective_config.json` binds the source checkpoint SHA/seed/step, target seed,
protocol and subset hashes, anchor payload, selected train/validation annotation,
calibration and disparity hashes, code manifest, environment, and update/selection
rules. `initialization.json` records the seeded model and reset-head tensor hashes
and the complete copied tensor list. Both training entrypoints set and record
the same explicit historical backend profile by default. The profile is based
on archived code and PyTorch 2.5 defaults; it is not a measurement of old runs.
`--backend-profile reproducible` disables TF32 and requests deterministic cuDNN
kernels. `--strict-determinism` selects that profile and also errors on
unsupported operations. Each choice belongs in a separate run configuration.

The output directory has an exclusive process lock. `last.pth` saves model,
optimizer, sampler epoch/cursor, completed updates, validation state, best step,
and Python/NumPy/Torch/CUDA RNG state. Validation caches retain a contiguous
prefix and are bound to the model tensor values, ordered rows, anchors and input
identity. A resumed run completes a pending validation before its next update.
No cached prefix or metric is reused across changed weights, inputs, or code.
Validation preserves the training RNG state. The dataset has no stochastic
augmentation; exact CPU resume tests do not promise bitwise equality across GPU
models or nondeterministic CUDA kernels.

`SIGUSR1`, `SIGTERM`, or the job time limit saves a checkpoint at a safe boundary
and returns **75**. For example, `--max-seconds 37800 --save-margin-seconds 900`
requests a checkpoint 900 seconds before a 10h30 scheduler job ends. A launcher
must use the actual remaining job time when the scheduler grants a shorter job;
`SLURM_JOB_END_TIME`, when supplied, is also honored as Unix seconds. Only the job
wrapper or operator resubmits; the runner creates no scheduler jobs or watcher.
`--allocation-updates N` tests resumption after N additional updates. Ordinary errors remain
failures. `training_complete.json` records the completed budget, all candidate
steps, selected step, and best/final checkpoint hashes.

`test.py` restores the checkpoint's backend settings before initializing CUDA,
uses its embedded anchors, and rejects a changed runtime or protocol. Detection
and depth use the same checkpoint and fixed keyframes. Its evaluation record
includes the effective backend settings and where they came from.
Legacy checkpoints use their matching original protocol/anchor files and are
marked as legacy in the evaluation record. If their backend flags were not
recorded, inference warns and records its assumption of the historical profile.
Their old `last.pth` files require
their original runner; the new runtime does not silently adopt them. Reuse of
historical results in a new comparison requires a separate equivalence audit.
Event HDF5/cache contents require their own preparation and a comparison of raw
and cached inputs; the per-job annotation/depth check does not hash all event bytes.

The CPU tests require the repository's PyTorch environment and no GPU or Waymo
installation:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s transfer/tests -v
```

They cover nested identities, excluded-label isolation and absent-class fallback,
source/head guards, embedded-anchor inference, and exact optimizer/RNG/sample
order equality across interrupted training and partial validation, plus backend
restoration from checkpoints. A separate GPU check must exercise DSEC
forward/backward and the official Waymo evaluator before full training.

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
run. The table and supplementary score above remain historical results; this
command alone does not establish equivalence to their original frozen runtime.

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
