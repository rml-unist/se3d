# SE3D dataset

SE3D was generated with CARLA 0.9.15 in eight towns (1–7 and 10) under six
combinations of illumination and rain. It has 58 sequences and 48,304 frames
recorded at 20 Hz; 46,229 frames contain at least one annotated object.

## Sensors

| Sensor | Specification |
|---|---|
| Stereo event cameras | 640×480, 90° horizontal FOV, f = 320 px, baseline 0.6 m, so disparity d = 192 / Z (Z in m) |
| Stereo RGB cameras | 1440×1080, at the event-camera poses |
| Dense disparity | left event view, valid for Z ≤ 100 m |
| LiDAR | 128 channels, vertical FOV −34.2° to +17.1° |
| Ego state | logged per frame (`coords.txt`, `speeds.txt`) |

Events come from the CARLA DVS sensor. The positive and negative contrast
thresholds are both 0.10 in daytime sunny and all nighttime scenes, 0.05 in
daytime rain and 0.01 in daytime heavy rain.

## Download and layout

The dataset is split into one archive per sequence, `SE3D_<sequence>.tar.gz`,
plus `SE3D_meta.tar.gz`. Every archive unpacks into `SE3D/`:

```bash
for f in SE3D_*.tar.gz; do tar xzf "$f"; done
sha256sum -c SHA256SUMS          # optional
python tools/check_dataset.py --data-root SE3D
```

```
SE3D/
├── calib.txt                       # shared by all sequences
├── label_correction_manifest.csv   # per-file record of the label deduplication
├── car_correction_decisions.json   # the 331 nested Car rows that were removed
└── map1/
    └── map1_day_sunny_moving/
        ├── timestamps.txt          # frame timestamps (CARLA simulation time, ns), one line per frame
        ├── events/{left,right}/events.h5, rectify_map.h5
        ├── disparity/event/<frame>.npy, disparity/timestamps_with_label.txt
        ├── label/<frame>.txt           # deduplicated 3D boxes (use these)
        ├── label_original/<frame>.txt  # boxes before deduplication
        ├── image_2/, image_3/      # left/right RGB, PNG
        ├── depth_map/              # depth at the RGB resolution, 16-bit PNG
        ├── dvs_2/, dvs_3/          # CARLA DVS renderings for visualization
        ├── velodyne/<frame>.bin    # LiDAR points, float32 (x, y, z, intensity)
        ├── coords.txt, speeds.txt  # ego state per frame
```

Sequence folders are named `map<town>_<day|night>_<sunny|rain|heavyrain>_moving`,
where a `_1` suffix marks a second sequence recorded in the same town and
condition. Per-frame files are named by frame number; in sorted order, the i-th
file of each folder belongs to the i-th line of `timestamps.txt`.

## File formats

- **events.h5**: `events/x`, `events/y` (uint16), `events/t` (int64, ns),
  `events/p` (bool, true for positive), and `ms_to_idx`, the index of the first
  event of each millisecond. `rectify_map.h5` holds the 480×640×2 rectification
  map applied when events are read. The files are uncompressed HDF5.
- **disparity/event/*.npy**: float32, 480×640, in pixels of the left event
  camera; values ≤ 0 mark pixels without ground truth (depth beyond 100 m).
- **label/*.txt**: KITTI format, 15 values per line: class, truncation,
  occlusion level, alpha, 2D box (x1 y1 x2 y2) in the 640×480 left event image,
  height, width, length (m), bottom-center location x y z (m) in the left event
  camera frame, and rotation_y. Classes: Car, Pedestrian, Bicycle, Motorcycle,
  Truck, Van, Bus.
- **calib.txt**: KITTI style. P0/P1 are the left/right RGB cameras (f = 720 px),
  P2/P3 the left/right event cameras (f = 320 px); boxes use P2.

## Annotations

| Version | Folder | Boxes | Use |
|---|---|---:|---|
| Deduplicated | `label/` | 175,123 | all evaluations, recommended for training |
| Before deduplication | `label_original/` | 184,884 | training data of the reported 40-epoch models |

Deduplication removed 9,761 rows: 5,153 Bus rows exported twice for the same
vehicle, 4,277 Truck rows that duplicated Van actors, and 331 Car rows nested in
a larger box of the same parked car. The deduplicated set has Car 95,270,
Pedestrian 28,273, Bicycle 13,308, Motorcycle 21,448, Truck 7,394, Van 4,277 and
Bus 5,153 boxes.

Difficulty levels follow KITTI with SE3D thresholds and no truncation limit:

| Level | 2D box height | Occlusion |
|---|---|---|
| Easy | > 40 px | 0 |
| Moderate | > 25 px | ≤ 1 |
| Hard | > 15 px | ≤ 2 |

## Splits

`splits/se3d_splits.json` assigns the 58 sequences to 40 training, 9
validation and 9 test sequences and lists the frames of each split as
`[timestamp, frame]`. No sequence appears in two splits; a town can appear in
several splits under different conditions.

| Condition | Frames | Annotated | Train | Val. | Test | Test towns |
|---|---:|---:|---:|---:|---:|---|
| Day sunny | 8,007 | 7,775 | 4,157 | 825 | 998 | 3, 10 |
| Day rain | 7,970 | 7,724 | 4,593 | 947 | 499 | 2 |
| Day heavy rain | 8,000 | 7,656 | 2,854 | 2,245 | 998 | 4, 7 |
| Night sunny | 8,000 | 7,545 | 6,076 | 0 | 498 | 1 |
| Night rain | 8,385 | 8,056 | 5,267 | 975 | 499 | 5 |
| Night heavy rain | 7,942 | 7,473 | 3,849 | 1,717 | 499 | 6 |
| Total | 48,304 | 46,229 | 26,796 | 6,709 | 3,991 | |

- `train`, `val`: frames that contain at least one Car.
- `test`: every second frame of the test sequences, starting at the third,
  with no filter on the objects they contain.
- `test_car_filtered` (2,731 frames): the earlier test list, which kept only
  frames with a Car. It is provided for comparison with the original
  submission.

The validation split has no Bus instances and no nighttime sunny frames.
