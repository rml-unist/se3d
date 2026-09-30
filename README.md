# SE3D: A Synthetic Stereo Event Camera Dataset for 3D Perception

SE3D is a stereo event camera dataset generated with CARLA 0.9.15. Every frame
provides rectified stereo events with a 0.6-m baseline, dense disparity for all
valid pixels within 100 m, and 3D boxes of seven classes, together with stereo
RGB images and 128-channel LiDAR. The 58 sequences (48,304 frames at 20 Hz)
cover daytime and nighttime driving in sunny, rain and heavy-rain settings
across eight CARLA towns.

This repository contains the data splits, the evaluation code, the two joint
baselines (EMOD and DSGN-event) and the transfer experiments on DSEC-3DOD.

| Folder | Contents |
|---|---|
| `splits/` | sequence splits and the frame lists of every split |
| `anchors/` | per-class anchor sizes computed from the training boxes |
| `se3d/`, `tools/` | data loading, training, testing, evaluation and dataset tools |
| `emod/` | EMOD: SE-CFF stereo network with a DSGN-style 3D detection head |
| `dsgn_event/` | DSGN-event: the DSGN network adapted to 10-channel event stacks |
| `transfer/` | scratch vs. SE3D-initialized training on DSEC-3DOD ([transfer/README.md](transfer/README.md)) |
| `docs/DATASET.md` | sensors, file formats, annotations and splits |

## Download

Everything is in the [SE3D Google Drive folder](https://drive.google.com/drive/folders/1zwnqBDSj8OoYPkiBQ1F-BFCwXPKsUnXw):

- `SE3D_v2/`: the dataset, one archive per sequence (`SE3D_<sequence>.tar.gz`),
  `SE3D_meta.tar.gz` and `SHA256SUMS`. All archives unpack into `SE3D/`.
- `weights/`: the checkpoints listed [below](#checkpoints), with `SHA256SUMS`.

```bash
cd /data && for f in SE3D_*.tar.gz; do tar xzf "$f"; done
python tools/check_dataset.py --data-root /data/SE3D
```

The files in the top level of the folder (`train/val/test.tar.gz`,
`EMOD_day_160.pth`, `EMOD_night_135.pth`) belong to the 2024 release of the
original submission. They use a different layout and are not read by this code.

## Installation

Tested with Python 3.9, PyTorch 2.5.1 (CUDA 12.4) on NVIDIA A100 and RTX 3090 GPUs.
No CUDA extension has to be compiled.

```bash
conda create -n se3d python=3.9 -y && conda activate se3d
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`docker/Dockerfile` builds the same environment.

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

## Baselines

Moderate AP40 (%) and disparity errors on the test frames. Both models were
trained for 40 epochs on all six conditions with the annotations before
deduplication (`label_original/`), and the checkpoint was selected on validation
mAP with 11 recall positions.

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
Sun: EMOD trained on the two sunny conditions; DSGN: DSGN-event):

| Condition | Cars | All | Sun | DSGN | MAE All | MAE Sun | MAE DSGN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day sunny | 1,080 | 35.09 | 25.42 | 64.78 | 1.516 | 0.990 | 1.424 |
| Day rain | 602 | 53.59 | 56.68 | 93.63 | 1.516 | 1.154 | 1.563 |
| Day heavy rain | 223 | 15.50 | 15.90 | 31.20 | 1.415 | 1.594 | 1.781 |
| Night sunny | 148 | 62.86 | 40.27 | 79.24 | 1.877 | 1.624 | 1.944 |
| Night rain | 7 | 0.00 | 0.00 | 0.00 | 1.695 | 1.864 | 1.504 |
| Night heavy rain | 0 | – | – | – | 0.755 | 0.995 | 1.185 |

Each condition is tested in one or two towns (`docs/DATASET.md`), so these
numbers also reflect the town and its traffic.

### Test a checkpoint

```bash
python tools/test.py --model dsgn_event --checkpoint weights/se3d_dsgn_event_40ep.pth \
    --data-root /data/SE3D --output results/dsgn_event
python tools/test.py --model emod --checkpoint weights/se3d_emod_40ep.pth \
    --data-root /data/SE3D --output results/emod
```

`results/<name>/metrics.json` holds the full report and `predictions.pkl` the
per-frame boxes (`--save-kitti` also writes KITTI files). The anchor sizes used
in training are selected from the checkpoint. On an RTX 3090 these commands
give the table values within 0.02 AP (GPU arithmetic differs slightly from the
A100 used for the paper); the disparity metrics agree to the third decimal.

### Train

```bash
# the reported models (annotations before deduplication)
python tools/train.py --model emod --data-root /data/SE3D --labels original --output runs/emod
python tools/train.py --model dsgn_event --data-root /data/SE3D --labels original --output runs/dsgn_event
# sunny-only EMOD (105 epochs, about the same number of updates as 40 epochs on all conditions)
python tools/train.py --model emod --data-root /data/SE3D --labels original --anchors original_sunny \
    --conditions day_sunny night_sunny --epochs 105 --output runs/emod_sunny
# new models: deduplicated annotations (default)
python tools/train.py --model dsgn_event --data-root /data/SE3D --output runs/dsgn_event_label
```

Training uses batch size 1, Adam with a learning rate and weight decay of 1e-4,
no augmentation, and the loss 0.5 × depth + 0.5 × detection. After each epoch
the model is evaluated on the validation split; `best.pth` is the epoch with the
highest Moderate mAP (AP11) and is used for both tasks. An interrupted run
resumes from `last.pth` when the command is repeated. One epoch has 26,796
updates. Both models fit on a 24-GB GPU (peak allocated memory about 14 GB).

Event stacks are computed from `events.h5` the first time a frame is read and
stored next to the events (about 3 MB per frame). `tools/prepare_event_cache.py`
builds them in advance with several processes.

## Transfer to DSEC-3DOD

[transfer/README.md](transfer/README.md) describes the data preparation,
training and evaluation. Each model is trained on DSEC-3DOD twice, from random
initialization and from an SE3D checkpoint, with the same target data, a
16-epoch schedule and the same seed. Waymo Level-2 AP (%) on the 1,178 test
keyframes and disparity MAE (px):

| Model | Init. | Vehicle | Ped. | V/P | MAE |
|---|---|---:|---:|---:|---:|
| DSGN-event (3 seeds) | Scratch | 5.38 ± 0.61 | 0.46 ± 0.23 | 2.92 ± 0.20 | 0.874 ± 0.011 |
| | SE3D | 8.26 ± 1.10 | 2.01 ± 0.95 | 5.13 ± 0.51 | 0.829 ± 0.017 |
| EMOD | Scratch | 0.38 | 0.09 | 0.24 | 1.039 |
| | SE3D | 2.16 | 0.01 | 1.08 | 0.813 |
| SE-CFF (depth only) | Scratch | – | – | – | 0.858 |
| | SE3D | – | – | – | 0.743 |

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

## Citation

If you use SE3D, please cite the SE3D paper by J. Shin, H. Jang, J. Song, S. Lee,
C. Ha, K. Joo and J. Jeon. The BibTeX entry will be added on publication.

## License

The code in this repository is released under the MIT license (`LICENSE`).
Third-party code keeps its original license; see `NOTICE.md`.
