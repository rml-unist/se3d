### SE3D: A Stereo Event Camera Dataset for 3D Perception in Autonomous Driving Across Diverse Weather Conditions

If you use any of this code, please cite both following publications:

### Maintainers
* [Jaechan Shin]()
* [Hyeon woo Jang]()
* [Jeonghwan Song]()

## Table of contents
- [Pre-requisite](#pre-requisite)
    * [Hardware](#hardware)
    * [Software](#software)
    * [Dataset](#dataset)
- [Getting started](#getting-started)
- [Training](#training)
- [Inference](#inference)
    * [Pre-trained model](#pre-trained-model)
- [Related publications](#related-publications)
- [License](#license)

## Pre-requisite
The following sections list the requirements for training/evaluation the model.

### Hardware
Tested on:
- **CPU** - AMD EPYC 7742 64-Core Processor
- **RAM** - 1 TB
- **GPU** - 2 x NVIDIA A100 (40 GB)

### Software
Tested on:
- [Ubuntu 20.04 and 18.04](https://ubuntu.com/)
- [NVIDIA Driver 550](https://www.nvidia.com/Download/index.aspx)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Dataset
Download [SE3D]() datasets.

#### 📂 Data structure
Our folder structure is as follows:
```
SE3D
├── map1
│   ├── map1_day_sunny_moving
│   │   ├── calib
│   │   │   ├── calib.txt
│   │   ├── disparity
│   │   │   ├── event
│   │   │   │   ├── 000000.npy
│   │   │   │   ├── ...
│   │   │   │   └── 000999.npy
│   │   │   └── timestamps_with_label.txt
│   │   └── events
│   │       ├── left
│   │       │   ├── events.h5
│   │       │   └── rectify_map.h5
│   │       └── right
│   │           ├── events.h5
│   │           └── rectify_map.h5
│   ├── ...
│   └── map1_night_heavyrain_moving               # same structure as map1/map1_day_sunny_moving
└── map2                                          # same structure as map1
└── ...
```

## Getting started

### Pull docker image
```bash
docker pull jcshinrml/se-od
```

### Run docker container
```bash
docker run \
    -v /path/to/code:/workspace/code \
    -v /path/to/data:/workspace/data \
    -it --gpus=all --ipc=host \
    jcshinrml/se-od
```

## Single-GPU Training
```bash
cd /workspace/code
bash python ./src/main.py
```

## Training
```bash
cd /workspace/code/scripts && bash distributed_main.sh
```

## Inference
```bash
cd /workspace/code/scripts && bash inference.sh
```

### Pre-trained model
:gear: You can download pre-trained model from [here]()

## Related publications

- [Stereo Depth from Events Cameras: Concentrate and Focus on the Future - CVPR 2022 (PDF)](https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Stereo_Depth_From_Events_Cameras_Concentrate_and_Focus_on_the_CVPR_2022_paper.pdf)

- [DSGN: Deep Stereo Geometry Network for 3D Object Detection - CVPR 2020 (PDF)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_DSGN_Deep_Stereo_Geometry_Network_for_3D_Object_Detection_CVPR_2020_paper.pdf)

## License

MIT license.
