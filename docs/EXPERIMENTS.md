# Training protocols

This page covers the historical training commands, corrected-label experiments
and the 8-epoch annotation comparison. Results for the corrected-label
40-epoch experiments are pending.

## Historical paper protocol (Tables IV and V)

Table IV uses original annotations for both training and validation, 40
complete passes over all training conditions, and Moderate 3D AP11 validation
selection. Table V's All and DSGN columns reuse those models. Its Sun model
uses 105 complete passes over 10,233 sunny training frames and validates on
the sunny subset after each pass, also with original annotations and AP11.
Test evaluation uses corrected `label/` annotations and AP40 for every model.

```bash
python tools/train.py --model emod --data-root /data/SE3D \
    --labels label_original --validation-labels label_original --selection-metric ap11 \
    --backend-profile historical --epochs 40 --seed 20260909 --output runs/hist_emod
python tools/train.py --model dsgn_event --data-root /data/SE3D \
    --labels label_original --validation-labels label_original --selection-metric ap11 \
    --backend-profile historical --epochs 40 --seed 20260909 --output runs/hist_dsgn_event
python tools/train.py --model emod --data-root /data/SE3D \
    --labels label_original --validation-labels label_original --selection-metric ap11 \
    --conditions day_sunny night_sunny --validation-conditions day_sunny night_sunny \
    --anchors label_original_sunny --backend-profile historical \
    --epochs 105 --seed 20260909 --output runs/hist_emod_sunny
```

These commands match the recorded data, budget and selection rules.
The `historical` backend profile uses the PyTorch 2.5 defaults of the archived
code. The original effective settings were not fully recorded, so retraining
may produce different weights and scores. Use the published checkpoints to
evaluate the historical models.

## Source training and model selection

Corrected-label runs use `label/` for training, validation and test. Anchors
are computed only from the corresponding training frames and annotations.
The existing split is fixed at 26,796 training, 6,709 validation and 3,991 test
frames. The seven class definitions, IoUs and difficulty thresholds are fixed.

Every run uses 1,071,840 optimizer updates, batch size 1, Adam with learning
rate and weight decay 1e-4, no augmentation, and loss weights 0.5 for disparity
and 0.5 for detection. The learning-rate schedule is constant, and each seed
starts from random initialization.

Validate every 26,796 updates on the same full, corrected validation split:
40 selection opportunities per run. Keep the checkpoint with the highest
Moderate 3D mAP40, with the earliest update winning ties. Use this checkpoint
for both test detection and disparity. Fixed-update weights at 214,368 and
1,071,840 updates are also saved independently of selection.

Validation mAP averages six classes because Bus has no validation ground truth;
test mAP averages seven. The [split table](DATASET.md#splits) lists condition
coverage, including the absence of night-sunny validation frames.

| Training arm | Seeds | Training frames per pass | Updates | Validation |
|---|---|---:|---:|---|
| EMOD, release labels, all conditions | 20260909/10/11 | 26,796 | 1,071,840 | corrected, full split, AP40 |
| DSGN-event, release labels, all conditions | 20260909/10/11 | 26,796 | 1,071,840 | corrected, full split, AP40 |
| EMOD, release labels, sunny training only | 20260909/10/11 | 10,233 | 1,071,840 | corrected, full split, AP40 |
| EMOD, original-label comparison | 20260909 | 26,796 | 1,071,840 | corrected, full split, AP40 |
| DSGN-event, original-label comparison | 20260909 | 26,796 | 1,071,840 | corrected, full split, AP40 |

Sunny training uses sunny-training anchors and 104 complete passes plus 7,608
updates of the next pass. Both arms use the same full-condition validation
split, including non-sunny labels for model selection.

The original-label controls use original-training anchors and corrected
validation. They measure the combined annotation-and-anchor change at one
paired seed.

```bash
python tools/train.py --model dsgn_event --data-root /data/SE3D \
    --labels label --validation-labels label --selection-metric ap40 \
    --updates 1071840 --validate-every 26796 --backend-profile historical --seed 20260909 \
    --output runs/dsgn_event_s20260909
python tools/train.py --model emod --data-root /data/SE3D \
    --conditions day_sunny night_sunny --updates 1071840 --validate-every 26796 \
    --backend-profile historical --seed 20260909 --output runs/emod_sunny_s20260909
```

Repeat primary runs with seeds 20260910 and 20260911. Summarize each model and
training arm separately using all three seeds: individual results, mean and
sample standard deviation (`ddof=1`), including per-class AP and disparity by
condition.

## Backend settings and determinism

Source and transfer training default to `--backend-profile historical`.
Use the same profile across the runs being compared.

| Setting | `historical` (default) | `reproducible` | `reproducible --strict-determinism` |
|---|---|---|---|
| cuDNN deterministic | false | true | true |
| cuDNN benchmark | false | false | false |
| cuDNN TF32 | true | false | false |
| Matrix-multiply TF32 | false | false | false |
| Error on nondeterministic operations | false | false | true |

Python, NumPy, CPU and CUDA RNGs are seeded in every profile. The reproducible
profile also supplies `CUBLAS_WORKSPACE_CONFIG=:4096:8` if no value is set.
The historical profile preserves an existing workspace environment variable;
its effective value is recorded. Strict mode requires `:4096:8` or `:16:8`.
`--strict-determinism` alone selects the reproducible profile; combining it
with an explicit historical profile is rejected.
`NVIDIA_TF32_OVERRIDE` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` are also recorded.
Conflicting overrides are rejected; inference requires them to match the
checkpoint's environment.

Neither profile guarantees bitwise-identical GPU training. Strict mode raises
an error for unsupported operations, including CUDA grid-sampling backward
used by these models, so it can stop training. These limits follow PyTorch's
[reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst)
and [TF32 settings](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst).

Checkpoints store the effective profile and flags, which test entrypoints
restore before CUDA initialization. Older checkpoints without these flags
use the historical profile with a warning. Evaluation output records the
settings and their source. Changing profiles requires a new output directory.

## Resuming training

Repeat the same command and output directory to resume from `last.pth`.
Use a new output directory when changing code, inputs or training settings.
See `python tools/train.py --help` for checkpoint and time-limit options.

## Earlier annotation comparison

The 8-epoch comparison evaluated four models on the same 3,991 test frames,
at seed 20260909 and 214,368 updates. Both annotations and training anchors
change together. Full values and checkpoint hashes are in
[`benchmarks/epoch8_label_comparison.json`](benchmarks/epoch8_label_comparison.json).

| Model | Training annotations | mAP40 (%) | Disparity MAE (px) |
|---|---|---:|---:|
| EMOD | original | 9.6122 | 1.1918 |
| EMOD | release | 9.4903 | 1.1628 |
| DSGN-event | original | 25.2492 | 1.5481 |
| DSGN-event | release | 23.0085 | 1.5303 |

Truck AP40 is zero in all four runs.
