# SE3D: A Synthetic Stereo Event Camera Dataset for 3D Perception

SE3D is a stereo event camera dataset generated with CARLA 0.9.15. Every frame
provides rectified stereo events with a 0.6-m baseline, dense disparity for all
valid pixels within 100 m, and 3D boxes of seven classes, together with stereo
RGB images and 128-channel LiDAR. The 58 sequences (48,304 frames at 20 Hz)
cover daytime and nighttime driving in sunny, rain and heavy-rain settings
across eight CARLA towns.

This repository contains the data splits, the evaluation code, the two joint
baselines (EMOD and DSGN-event) and the transfer experiments on DSEC-3DOD.
The tables below report the manuscript's historical experiments.
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) documents those protocols and
training with corrected annotations and AP40 model selection.

| Folder | Contents |
|---|---|
| `splits/` | sequence splits and the frame lists of every split |
| `anchors/` | per-class anchor sizes computed from the training boxes |
| `se3d/`, `tools/` | data loading, training, testing, evaluation and dataset tools |
| `emod/` | EMOD: SE-CFF stereo network with a DSGN-style 3D detection head |
| `dsgn_event/` | DSGN-event: the DSGN network adapted to 10-channel event stacks |
| `transfer/` | scratch vs. SE3D-initialized training on DSEC-3DOD ([transfer/README.md](transfer/README.md)) |
| `docs/DATASET.md` | sensors, file formats, annotations and splits |
| `docs/EXPERIMENTS.md` | annotation versions, model selection, seeds and experiment budgets |

## Download

Use the [SE3D_v2 dataset folder](https://drive.google.com/drive/folders/1zNbe4N9Ash-lirVgAW3SUeYnkAV4aExY)
and the [weights folder](https://drive.google.com/drive/folders/1MwzAA26ub8axup1DAWub370x7JyHbtHl):

- `SE3D_v2/`: the dataset, one archive per sequence (`SE3D_<sequence>.tar.gz`),
  `SE3D_meta.tar.gz` and `SHA256SUMS`. All archives unpack into `SE3D/`.
- `weights/`: the checkpoints listed [below](#checkpoints), with `SHA256SUMS`.

If you downloaded the dataset before **2026-10-10**, update and extract
`SE3D_meta.tar.gz` for the license and annotation metadata, and replace
`manifest.json` and `SHA256SUMS`. The sequence archives are unchanged.

The complete dataset download is **432.6 GB** (decimal GB), across 58 sequence
archives and one metadata archive; extraction occupies about **1.06 TB**.
Download individual archives to retry or
resume an interrupted transfer. Check their hashes before extraction. See
[storage and caches](docs/DATASET.md#storage-and-event-caches) for extracted
size and the additional event-cache requirement.

```bash
cd /data
sha256sum -c --ignore-missing SHA256SUMS
for f in SE3D_*.tar.gz; do tar xzf "$f"; done
python tools/check_dataset.py --data-root /data/SE3D
```

`--ignore-missing` checks the archives you downloaded, so it also works for
the two-file quick check below. It does not confirm that the whole dataset is
present; `tools/check_dataset.py` checks the complete extracted dataset.

The files in the top level of the [older download folder](https://drive.google.com/drive/folders/1zwnqBDSj8OoYPkiBQ1F-BFCwXPKsUnXw) (`train/val/test.tar.gz`,
`EMOD_day_160.pth`, `EMOD_night_135.pth`) belong to the 2024 release of the
original submission. They use a different layout and are not read by this code.

## Installation

Tested with Python 3.9/3.10, PyTorch 2.5.1 (CUDA 12.4) on NVIDIA A100 and RTX 3090 GPUs.
No CUDA extension has to be compiled.

```bash
conda create -n se3d python=3.9 -y && conda activate se3d
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`docker/Dockerfile` builds the same environment.

For a small GPU check, download only `SE3D_meta.tar.gz` and
`SE3D_map3_day_sunny_moving.tar.gz`, extract them, and run:

```bash
python tools/quick_check.py --data-root /data/SE3D --model dsgn_event
# Optional: add --checkpoint weights/se3d_dsgn_event_40ep.pth
```

This runs inference on two test frames. `python tools/check_protocol.py`
checks the split and anchor files without data or a GPU.

## Benchmark

Both tasks are scored on the 3,991 test frames with the deduplicated annotations (`label/`).

- **Detection:** Moderate 3D AP with 40 recall positions (AP40), averaged over
  the seven classes. IoU thresholds are 0.7 for Car, Truck, Van and Bus and 0.5
  for Pedestrian, Bicycle and Motorcycle. Difficulty levels use 2D box heights
  above 40/25/15 px and occlusion up to 0/1/2, without a truncation limit.
  BEV AP and AP11 are reported as well.
- **Disparity:** MAE, RMSE and the percentage of pixels with an error above 1
  and 2 px (1PE, 2PE), over pixels with a valid ground truth.

To score another method, write one KITTI-format file per test frame (with the
score in the 16th column) and, optionally, one 480×640 float32 disparity map:

```bash
python tools/evaluate.py --data-root /data/SE3D \
    --kitti-dir my_method/kitti --disparity-dir my_method/disparity --output my_method/metrics.json
```

Files are named `<map>/<sequence>/<frame>.txt` and `<map>/<sequence>/<frame>.npy`;
the frames are listed under `"test"` in `splits/se3d_splits.json`. The output
contains overall and per-condition results for Easy, Moderate and Hard.

## Historical baselines

Moderate AP40 (%) and disparity errors on the test frames. Both models were
trained for 40 epochs on all six conditions with the annotations before
deduplication (`label_original/`), and the checkpoint was selected on validation
mAP with 11 recall positions using the original validation annotations. These
are the current manuscript's Table IV results, not corrected-label retraining.

| Class | GT | EMOD 3D | EMOD BEV | DSGN-event 3D | DSGN-event BEV |
|---|---:|---:|---:|---:|---:|
| Car | 2,060 | 38.68 | 50.64 | 68.75 | 80.21 |
| Pedestrian | 379 | 0.00 | 0.00 | 0.94 | 1.02 |
| Bicycle | 515 | 0.73 | 1.05 | 2.69 | 4.61 |
| Motorcycle | 450 | 7.85 | 8.51 | 34.44 | 43.26 |
| Truck | 88 | 0.00 | 0.00 | 0.00 | 0.00 |
| Van | 55 | 26.31 | 27.24 | 80.47 | 80.47 |
| Bus | 975 | 0.03 | 0.03 | 0.66 | 1.00 |
| **mAP** | | **10.52** | **12.50** | **26.85** | **30.08** |
| Disparity MAE / RMSE (px) | | 1.460 / 11.798 | | 1.577 / 11.708 | |
| 1PE / 2PE (%) | | 20.69 / 12.37 | | 28.96 / 14.70 | |

Car AP40 and disparity MAE per condition (All: EMOD trained on all conditions;
Sun: EMOD trained on the two sunny conditions; DSGN: DSGN-event), corresponding
to manuscript Table V:

| Condition | Cars | All | Sun | DSGN | MAE All | MAE Sun | MAE DSGN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day sunny | 1,080 | 35.09 | 25.42 | 64.78 | 1.516 | 0.990 | 1.424 |
| Day rain | 602 | 53.59 | 56.68 | 93.63 | 1.516 | 1.154 | 1.563 |
| Day heavy rain | 223 | 15.50 | 15.90 | 31.20 | 1.415 | 1.594 | 1.781 |
| Night sunny | 148 | 62.86 | 40.27 | 79.24 | 1.877 | 1.624 | 1.944 |
| Night rain | 7 | 0.00 | 0.00 | 0.00 | 1.695 | 1.864 | 1.504 |
| Night heavy rain | 0 | – | – | – | 0.755 | 0.995 | 1.185 |

Each condition is tested in one or two towns (`docs/DATASET.md`), so these
numbers also reflect the town and its traffic. Night-rain Car AP rests on only
seven boxes; night-heavy-rain has no Car GT and its AP is undefined (–).
The validation split contains neither Bus nor night-sunny examples, which
limits what checkpoint selection can measure. The split is kept fixed.
The historical Sun model used 105 passes over 10,233 sunny training frames
and AP11 selection on the sunny validation subset.

A matched **8-epoch, one-seed annotation comparison** is available in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#earlier-annotation-comparison).

### Test a checkpoint

```bash
python tools/test.py --model dsgn_event --checkpoint weights/se3d_dsgn_event_40ep.pth \
    --data-root /data/SE3D --output results/dsgn_event
python tools/test.py --model emod --checkpoint weights/se3d_emod_40ep.pth \
    --data-root /data/SE3D --output results/emod
```

`results/<name>/metrics.json` holds the full report and `predictions.pkl` the
per-frame boxes (`--save-kitti` also writes KITTI files). Anchor sizes and
backend settings are loaded from the checkpoint.

### Train

To retrain the historical protocols in Tables IV and V:

```bash
# Table IV; the EMOD run is also Table V's All model
python tools/train.py --model emod --data-root /data/SE3D \
    --labels label_original --validation-labels label_original --selection-metric ap11 \
    --backend-profile historical --epochs 40 --seed 20260909 --output runs/hist_emod
python tools/train.py --model dsgn_event --data-root /data/SE3D \
    --labels label_original --validation-labels label_original --selection-metric ap11 \
    --backend-profile historical --epochs 40 --seed 20260909 --output runs/hist_dsgn_event
# Table V Sun: sunny training and sunny validation, 105 complete passes
python tools/train.py --model emod --data-root /data/SE3D \
    --labels label_original --validation-labels label_original --selection-metric ap11 \
    --conditions day_sunny night_sunny --validation-conditions day_sunny night_sunny \
    --anchors label_original_sunny --backend-profile historical \
    --epochs 105 --seed 20260909 --output runs/hist_emod_sunny
```

These commands match the historical data and selection rules. Retraining can
differ because the original backend settings were not fully recorded; see
[reproduction details](docs/EXPERIMENTS.md#historical-paper-protocol-tables-iv-and-v).

For training with corrected annotations and AP40 selection:

```bash
# Corrected labels for training and validation (the default)
python tools/train.py --model emod --data-root /data/SE3D \
    --backend-profile historical --seed 20260909 --output runs/emod_s20260909
python tools/train.py --model dsgn_event --data-root /data/SE3D \
    --backend-profile historical --seed 20260909 --output runs/dsgn_event_s20260909
# Sunny training: exactly the same update budget and full validation schedule
python tools/train.py --model emod --data-root /data/SE3D \
    --conditions day_sunny night_sunny --updates 1071840 --validate-every 26796 \
    --backend-profile historical --seed 20260909 --output runs/emod_sunny_s20260909
```

These runs use batch size 1, Adam with learning rate and weight decay 1e-4,
no augmentation, and loss 0.5 × depth + 0.5 × detection. They train for
1,071,840 updates and validate every 26,796 updates. `best.pth` has the highest
validation Moderate 3D mAP40 over classes with ground truth; ties keep the
earliest update. Use it for both test tasks. Repeating an interrupted command
resumes from `last.pth`. The [full protocol](docs/EXPERIMENTS.md) lists seeds,
comparison runs and saved intermediate checkpoints.

`--labels label|label_original` selects the training annotations. Validation
defaults to `label`; change it with `--validation-labels` and select AP11 or
AP40 with `--selection-metric`.

Source and transfer training share `--backend-profile historical|reproducible`
(default: `historical`). See the
[backend settings table](docs/EXPERIMENTS.md#backend-settings-and-determinism)
for TF32 and determinism options.

Both models fit on a 24-GB GPU (historical peak allocated memory about 14 GB).
Historical A100 runs took about 31.9 GPU-hours for EMOD and 30.0 for DSGN-event
at eight epochs, including startup, data loading and validation. Linear scaling
estimates about 160 and 150 GPU-hours for 40 epochs, excluding queue time.
These estimates are not upper bounds and apply to the historical profile;
`reproducible` and strict mode may be slower and have not been timed.

Event stacks are computed from `events.h5` the first time a frame is read and
stored next to the events (about 3 MB per frame). `tools/prepare_event_cache.py`
builds them in advance with several processes. All three main splits need
roughly **110 GB** of extra cache storage. Cache creation needs write access;
pass `--cache-root /scratch/se3d-cache` to preparation, training and testing to
keep the dataset itself read-only. Existing complete caches can be read in place.

## Transfer to DSEC-3DOD

[transfer/README.md](transfer/README.md) describes the data preparation,
training and evaluation. Each model is trained on DSEC-3DOD twice, from random
initialization and from an SE3D checkpoint, with the same target data, a
16-epoch schedule and the same seed. Waymo Level-2 AP (%) on the 1,178 test
keyframes and disparity MAE (px), corresponding to manuscript Table VI:

| Model | Init. | Vehicle | Ped. | V/P | MAE |
|---|---|---:|---:|---:|---:|
| DSGN-event (3 seeds) | Scratch | 5.38 ± 0.61 | 0.46 ± 0.23 | 2.92 ± 0.20 | 0.874 ± 0.011 |
| | SE3D | 8.26 ± 1.10 | 2.01 ± 0.95 | 5.13 ± 0.51 | 0.829 ± 0.017 |
| EMOD | Scratch | 0.38 | 0.09 | 0.24 | 1.039 |
| | SE3D | 2.16 | 0.01 | 1.08 | 0.813 |
| SE-CFF (depth only) | Scratch | – | – | – | 0.858 |
| | SE3D | – | – | – | 0.743 |

These historical transfer runs use the original-label source checkpoints.
The DSGN-event mean ± sample SD varies three **target** seeds while fixing
one source model. Corrected-source, source-seed and label-efficiency follow-ups
are specified in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#transfer-follow-ups).

## Checkpoints

The [weights folder](https://drive.google.com/drive/folders/1MwzAA26ub8axup1DAWub370x7JyHbtHl)
contains four checkpoints (314 MB): two for the SE3D benchmark and two source
checkpoints for transfer learning. SHA256 sums are in `weights/SHA256SUMS`.

| File | Model | Training |
|---|---|---|
| `se3d_emod_40ep.pth` | EMOD | SE3D, 40 epochs, all conditions (selected epoch 17) |
| `se3d_dsgn_event_40ep.pth` | DSGN-event | SE3D, 40 epochs, all conditions (selected epoch 29) |
| `se3d_emod_8ep.pth` | EMOD | SE3D, 8 epochs; initialization for the EMOD and SE-CFF transfer |
| `se3d_dsgn_event_8ep.pth` | DSGN-event | SE3D, 8 epochs; initialization for the DSGN-event transfer |

These four published files were trained with `label_original/`. The historical
sunny-only and corrected-label checkpoints are not in that public folder.

## Citation

If you use SE3D, please cite the SE3D paper by J. Shin, H. Jang, J. Song, S. Lee,
C. Ha, K. Joo and J. Jeon. The manuscript is under review; a public paper link
and BibTeX entry will be added when available.

## License

The code is released under the [MIT license](LICENSE). The SE3D authors' rights
in the sensor data, annotations, calibration and splits are also released
under MIT; see the [dataset license](docs/DATASET_LICENSE.md). CARLA assets keep
their upstream terms and [attribution notice](docs/DATASET_NOTICE.md).
Third-party code keeps its original license; see [NOTICE.md](NOTICE.md).
