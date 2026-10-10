# Release experiment protocol

The manuscript is under review. The existing benchmark tables describe its
historical runs. The experiments below are a new, predeclared release study;
their results and checkpoints must not be described as complete before the
training, evaluation and receipts have been verified.

## Source training and model selection

All new primary runs use `label/` for training, validation and test. Anchors
are computed only from the corresponding training frames and annotations.
The existing split is fixed at 26,796 training, 6,709 validation and 3,991 test
frames. The seven class definitions, IoUs and difficulty thresholds are fixed.

Every run uses 1,071,840 optimizer updates, batch size 1, Adam with learning
rate and weight decay 1e-4, no augmentation, and loss weights 0.5 for disparity
and 0.5 for detection. The learning-rate schedule is constant. Training starts
from random initialization for each seed; a previous 8-epoch model is not
continued and described as a fresh run.

Validate every 26,796 updates on the same full, corrected validation split:
40 selection opportunities per run. Keep the checkpoint with the highest
Moderate 3D mAP40, with the earliest update winning ties. Use this checkpoint
for both test detection and disparity. Fixed-update weights at 214,368 and
1,071,840 updates are also saved independently of selection.

Validation mAP averages the six classes with valid Moderate GT; Bus has no
validation GT. Validation also has no night-sunny sequence. The full test has
seven evaluable classes. The metric definition is shared, but validation and
test means have different class support and are not interchangeable scores.

| Training arm | Seeds | Training frames per pass | Updates | Validation |
|---|---|---:|---:|---|
| EMOD, release labels, all conditions | 20260909/10/11 | 26,796 | 1,071,840 | corrected, full split, AP40 |
| DSGN-event, release labels, all conditions | 20260909/10/11 | 26,796 | 1,071,840 | corrected, full split, AP40 |
| EMOD, release labels, sunny training only | 20260909/10/11 | 10,233 | 1,071,840 | corrected, full split, AP40 |
| EMOD, original-label comparison | 20260909 | 26,796 | 1,071,840 | corrected, full split, AP40 |
| DSGN-event, original-label comparison | 20260909 | 26,796 | 1,071,840 | corrected, full split, AP40 |

Sunny training uses 104 complete passes plus 7,608 updates of the next pass.
It has the same update budget and validation opportunities as all-condition
training. It uses sunny-training anchors. Full-condition validation is shared
by both arms; this is a comparison of **training conditions**, not a claim that
non-sunny labeled data were never available for model selection.

The original-label controls use original-training anchors and corrected
validation. They measure the combined annotation-and-anchor change at one
paired seed. Historical 40-epoch models selected using original-label AP11
are separate references, not a controlled substitute for these runs.

```bash
python tools/train.py --model dsgn_event --data-root /data/SE3D \
    --labels label --validation-labels label --selection-metric ap40 \
    --updates 1071840 --validate-every 26796 --seed 20260909 \
    --output runs/dsgn_event_s20260909
python tools/train.py --model emod --data-root /data/SE3D \
    --conditions day_sunny night_sunny --updates 1071840 --validate-every 26796 \
    --seed 20260909 --output runs/emod_sunny_s20260909
```

Repeat primary runs with seeds 20260910 and 20260911. Report all per-seed
results and their mean and **sample** standard deviation (`ddof=1`), including
per-class AP and condition-specific disparity results. Do not select a seed
using test scores or combine different source/selection protocols into one
mean. Three-seed SD describes these runs; it is not a guaranteed AP tolerance.

## Runtime and resumption

Python, NumPy, CPU and CUDA RNGs are seeded. cuDNN benchmarking and TF32 are
disabled and cuDNN deterministic algorithms are requested. This does not
guarantee that every CUDA operation is deterministic: `--strict-determinism`
requests an error for unsupported operations. Record this setting rather than
claiming bit-for-bit repeatability across machines.

`effective_config.json` records annotation, split, anchor and runtime source
hashes, budget, selection rule and seed. `environment.json` records package,
CUDA and GPU information. `last.pth` stores optimizer and RNG states, epoch,
within-epoch cursor and completed validation state. Repeating the same command
resumes it; mismatched code, inputs or configuration are rejected.

`--max-seconds` and `--allocation-updates` limit one allocation without changing
the experiment budget. A time limit, SIGTERM or SIGUSR1 saves and returns exit
code 75. Validation interrupted at a boundary is rerun before further updates.
A scheduler may requeue exit 75; unexpected failures must remain failures.

## Earlier annotation comparison

The fixed 8-epoch study already evaluated four arms on the same 3,991 test
frames, at seed 20260909 and 214,368 updates. Both labels and their train-only
anchors change together. These are one-seed reference results, not the new
40-epoch study above. Full values and checkpoint hashes are in
[`benchmarks/epoch8_label_comparison.json`](benchmarks/epoch8_label_comparison.json).

| Model | Training annotations | mAP40 (%) | Disparity MAE (px) |
|---|---|---:|---:|
| EMOD | original | 9.6122 | 1.1918 |
| EMOD | release | 9.4903 | 1.1628 |
| DSGN-event | original | 25.2492 | 1.5481 |
| DSGN-event | release | 23.0085 | 1.5303 |

Truck AP40 is zero in all four arms. The study does not establish that label
duplication alone explains the weak Truck or Bus performance.

## Transfer follow-ups

The completed manuscript study varied target seeds while holding the original
8-epoch source fixed. New corrected-source experiments use fixed 214,368-update
source snapshots, distinguish source-seed from target-seed variation, and
preserve the 3,906/434/1,178 target split and official metric.

| Target study (DSGN-event) | Source seeds | Target seeds | Runs |
|---|---|---|---:|
| Full-label scratch | none | 20260909/10/11 | 3 |
| Full-label corrected-source transfer | 20260909/10/11 | 20260909/10/11, crossed with every source | 9 |
| Label efficiency, scratch and corrected-source pairs | fixed 20260909 | 20260909/10/11 at each fraction | 18 |

Each target run uses exactly 62,496 updates and validation every 3,906 updates
(16 selection slots). Select the highest official mean Vehicle/Pedestrian
Level-2 validation AP; the earliest tied step wins. Test once at that selected
checkpoint and use the same checkpoint for disparity evaluation.

Report all nine pretrained source/target cells. Target-seed variation at fixed
source 20260909 and source-seed variation at fixed target 20260909 answer
different questions. Do not count the nine correlated cells or reused scratch
references as nine independent paired trials. Existing target runs can be
reused only if their source hash, model/loss, initialization, data, anchors,
optimizer, budget, validation schedule and evaluator match the new protocol;
otherwise retain them as historical references.

The deferred label-efficiency experiment uses nested **training** chunks only,
with the same subsets, initialization rules, optimizer budget and selection
schedule in each scratch/pretrained pair. Both depth and box supervision are
restricted to the chosen subset. Anchors must be recomputed from that subset;
full-training annotation statistics would use excluded labels. Record actual
frame and object counts, not only a nominal percentage. See `transfer/` for
the frozen subset definitions and execution commands.

The subsets contain 13/32/63 of the 126 training chunks, corresponding to
403/992/1,953 keyframes (10.32%/25.40%/50.00%, nominally 10%/25%/50%). Chunks
are ranked once using SHA256 of the fixed salt
`se3d-target-label-efficiency-chunks-v1/20261010/` and the chunk name. Smaller
subsets are prefixes of larger ones and are shared across optimization seeds.
Both arms cycle through their selected frames to the same update budget; the
number of passes varies by fraction. Missing-class anchor defaults are fixed
before training and recorded with provenance, without inspecting excluded
target annotations. Held-out chunks are unchanged.

The complete study has at most 41 training runs before any verified reuse:
11 source, 12 full-label target and 18 label-efficiency runs. Historical timing
suggests roughly 1,870–2,220 GPU-hours including an allowance for gates and
final inference. This is a planning estimate, not a measured completion time.
Queue delays and available concurrency determine the calendar duration.
