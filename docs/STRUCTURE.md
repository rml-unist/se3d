# Runtime and maintenance layout

The public entry points are `tools/train.py`, `tools/test.py`,
`tools/evaluate.py` and the scripts in `transfer/`. Install the root
`requirements.txt` after PyTorch. There is no separate editable install for
`emod/` or `dsgn_event/`, and no local C++/CUDA extension to build.

| Component | Responsibility |
|---|---|
| `se3d/protocol.py` | canonical annotation names, split lists and condition names |
| `se3d/data.py` | explicit split frames, labels, sensor inputs and event cache location |
| `se3d/models.py` | baseline construction and shared class/anchor configuration |
| `se3d/engine.py` | inference, postprocessing and shared metric aggregation |
| `emod/configs/se3d_revision.py` | one implementation of SE3D and DSEC class/anchor configuration |
| `emod/src/lib/dsgn/` | detection head, training loss and postprocessing used by the joint baselines |
| `dsgn_event/dsgn/models/stereonet.py` | the distinct DSGN stereo backbone used by DSGN-event |
| `tools/maintainers/` | archive import, release packaging and completed-run verification |

The two DSGN-derived directories retain checkpoint parameter names and import
paths. They serve different model roles; the public DSGN-event wrapper uses
the shared detection loss and postprocessor from `emod/src/lib/dsgn/`.
Unused upstream extension build sources and machine-specific KITTI shell
launchers have been removed. Third-party provenance remains in `NOTICE.md`.

## Annotation conversion

Every public runtime command accepts `--labels label|label_original`. `label/`
is the release annotation set. `label_original/` is kept for explicitly named
historical comparisons. Older command names `corrected` and `original` are
accepted with a deprecation warning and normalized in recorded configuration.
Internal numbered label directories are not runtime inputs.

The maintainer importer reads the archived internal capture and the verified
annotation ZIP, then creates a separate public-layout working tree. It copies
both annotation sets and links large sensor files and existing caches. It
checks exact file lists, row format, archive checksum and dataset totals;
existing conflicting files cause an error. Raw captures are never renamed.

```bash
python tools/maintainers/import_legacy_dataset.py \
    --source /archive/internal-capture --annotations-archive /archive/annotations.zip \
    --output /scratch/SE3D
python tools/check_dataset.py --data-root /scratch/SE3D
python tools/maintainers/package_dataset.py --source /scratch/SE3D \
    --output /scratch/release --dry-run
python tools/maintainers/package_dataset.py --source /scratch/SE3D \
    --output /scratch/release
```

Packaging dereferences sensor links and includes the dataset MIT license.
Use a new output directory for a new release: old published archives and
checksums remain identifiable. `--meta-only` prepares the small metadata
archive without recompressing sensor data.

## Checks

`python tools/check_protocol.py` and `python -m unittest discover -s tests -v`
run without the dataset, PyTorch or a GPU. CI also checks Python syntax. For a
real input/inference check, use `tools/quick_check.py` as described in README.
The small check is not a benchmark and does not establish training variance.

For a completed new training run:

```bash
python tools/maintainers/verify_training_run.py --run runs/emod_s20260909
```

This checks the update budget, validation schedule, annotation/selection
metadata, fixed-update source weights, and earliest-tie checkpoint selection.
