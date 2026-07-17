# ResNet-18 two-epoch predictor comparison notebook

## Goal

Create a separate, portable Jupyter notebook that compares the repository's
EB-JEPA MLP projector with `FixedTreePredictor`. Both models use an identical
ResNet-18 backbone and train for exactly two epochs on CIFAR-10. The notebook
must run on either a local Mac or Google Colab and leave the existing workflow
notebook and YAML configurations unchanged.

This is a fast, controlled experiment. Its outputs compare behavior after the
same two-epoch budget; they do not establish converged model quality.

## Artifact and scope

The new artifact is:

`eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

The notebook is a reader-facing experiment report and executable workflow. It
uses the existing `comparison` package for configuration, model construction,
training, evaluation, checkpoint loading, parameter counting, and FLOP
analysis. Notebook-local orchestration may measure elapsed training time,
portable device latency, and format or export results. No production module
changes are required unless implementation reveals a small, independently
testable gap that cannot safely remain notebook-local.

The existing `scripts/workflow.ipynb`, `configs/baseline.yaml`, and
`configs/cortical.yaml` are not modified.

## Experiment protocol

The two experiment arms are:

1. **EB-JEPA:** ResNet-18 backbone plus the existing `MLPProjector` baseline
   head.
2. **Your predictor:** the same ResNet-18 backbone plus the existing
   `FixedTreePredictor` cortical head.

Both arms use:

- CIFAR-10 and the repository's existing split and paired-view augmentations;
- training seed `1`;
- batch size `256`;
- VICReg with the same coefficients;
- the same optimizer and learning-rate settings;
- exactly two epochs;
- zero warm-up epochs, because a ten-epoch warm-up cannot complete in a
  two-epoch experiment;
- five deterministic evaluation pair seeds `(11, 22, 33, 44, 55)`;
- sequential training to prevent competition for GPU or MPS memory.

The configurations are loaded from the existing baseline and cortical YAML
files, then normalized with notebook overrides. At minimum, overrides set
`model.backbone=resnet18`, `optimization.epochs=2`,
`optimization.warmup_epochs=0`, a shared valid learning rate, and the resolved
data root. Platform-specific worker and memory settings may also be overridden
without changing the scientific comparison.

Each arm writes to its own directory beneath a notebook-specific output root.
The notebook records the effective resolved configurations with its exported
results.

## Portable environment behavior

One setup cell detects Google Colab using import availability rather than a
hard-coded user path.

On Colab, the notebook locates an already-cloned repository under `/content`
or prints a concise setup instruction if it is absent. It adds the comparison
package's `src` directory to `sys.path` and uses an output root under the
project directory.

Locally, the notebook searches upward from the current working directory and
the notebook directory candidates for `eb_jepa_cifar10_comparison`. It rejects
ambiguous or missing roots with an actionable error. It must not contain a
`/Users/...` path.

Runtime device selection follows CUDA, then MPS, then CPU. The notebook labels
the selected device and records it in the exports. CIFAR-10 is reused from the
resolved data directory when present and downloaded through the existing data
pipeline when absent.

## Notebook structure and data flow

The notebook uses this top-to-bottom structure:

1. **Goal** — scope, interpretation limits, and a short protocol summary.
2. **Setup** — portable root discovery, imports, package checks, and device
   discovery.
3. **Parameters** — one editable cell containing the seed, epoch count, batch
   size, pair seeds, paths, and portable benchmark repetitions.
4. **Build configurations** — load both YAML files, apply shared overrides, and
   display the effective differences.
5. **Protocol checks** — assert an identical ResNet-18 backbone and shared
   training, loss, data, and evaluation settings; allow only the intended head
   and cortical-section differences.
6. **Train** — run EB-JEPA and then the predictor, time each call, and retain
   the best checkpoint and training summary.
7. **Evaluate** — evaluate each best checkpoint on the same five deterministic
   pair seeds and collect VICReg and collapse metrics.
8. **Compute profile** — reload checkpoints and measure trainable parameters,
   FLOPs, latency, and throughput for each head and complete model.
9. **Results** — produce comparison tables, focused plots, percentage changes,
   and concise evidence-based takeaways.
10. **Export and checks** — write CSV and JSON artifacts and assert that the
    results are finite, complete, and traceable to the expected checkpoints.

Long-running training and evaluation cells are visually separated so a reader
can distinguish setup from expensive execution.

## Comparison metrics

The results include:

- best validation VICReg loss;
- training wall-clock duration;
- mean and sample standard deviation of total test VICReg loss across five
  deterministic pair seeds;
- per-arm mean invariance, variance, and covariance components;
- minimum mean standard deviation, minimum feature standard deviation, and
  maximum collapsed fraction across evaluation pairs;
- trainable backbone, head, and total parameter counts;
- estimated head and full-model FLOPs per sample;
- head and full-model latency in milliseconds;
- head and full-model throughput in samples per second;
- absolute differences and percentage changes from EB-JEPA to the predictor.

Lower values are labeled as better only for loss, FLOPs, parameter count, and
latency. Higher values are labeled as better for throughput. Collapse metrics
are interpreted with the repository's existing thresholds rather than a
generic direction marker.

FLOPs use the existing `fvcore`-based analysis. Unsupported operators produce
a visible caveat and a missing FLOP value rather than silently reporting zero.

## Portable performance measurement

The repository's strict benchmark requires an NVIDIA A100, bfloat16, batch
size 256, 100 warm-ups, and 500 measurements. That protocol is not portable to
MPS, CPU, or general Colab runtimes, so this notebook does not call the strict
benchmark workflow.

Instead, a notebook-local profiler measures the active device. It uses a shared
batch size and repetition counts for both arms, synchronizes CUDA or MPS around
timed regions, performs warm-up calls, and uses a monotonic wall clock. The
table records the device, dtype, batch size, warm-ups, and measurements. Its
heading explicitly identifies the values as a local-device benchmark, not an
A100 protocol result.

## Presentation and exports

The notebook renders:

- a compact protocol table;
- a quality and collapse comparison table;
- a compute comparison table;
- a grouped quality-component chart;
- a compute-efficiency chart;
- a concise takeaway block derived only from collected values.

CSV exports contain flat comparison tables. A JSON export contains schema
version `1`, timestamp, environment metadata, effective configurations,
checkpoint paths and hashes, raw pair evaluations, training summaries, compute
metrics, and derived differences. Files are written beneath the
notebook-specific output root so other experiment artifacts are not replaced.

## Error handling

The notebook stops early with an actionable message when:

- the comparison project cannot be found;
- required imports are unavailable;
- either source YAML file is missing or invalid;
- CIFAR-10 cannot be found or downloaded;
- a training run fails to produce its best checkpoint;
- a checkpoint does not match its configured backbone or head;
- an evaluation does not return exactly five distinct pair results;
- a required result is non-finite;
- an output artifact cannot be written.

FLOP-analysis limitations are non-fatal only when they are clearly surfaced in
the results and JSON caveats. A device without CUDA or MPS falls back to CPU
with a warning that training may be slow.

## Validation

Implementation validation covers:

- notebook JSON and `nbformat` validity;
- required section order and stable cell identifiers;
- absence of hard-coded local user paths;
- explicit `resnet18`, two-epoch, zero-warm-up, and shared-seed settings;
- both baseline and cortical configuration paths;
- portable local and Colab root-discovery branches;
- syntax compilation of every code cell after filtering notebook magics;
- a lightweight configuration/model smoke test that builds both arms and
  verifies `(batch, 512)` backbone features and `(batch, 2048)` projections;
- existing relevant package tests.

The two full training runs are not executed as part of repository validation
because they are intentionally expensive. The handoff states this execution
gap and provides the exact command for executing the notebook top-to-bottom in
the prepared project environment.

## Acceptance criteria

The work is complete when the separate notebook is valid, portable, clearly
documents the two-epoch limitation, builds both ResNet-18 experiment arms with
identical shared settings, contains the full train/evaluate/profile/compare
workflow, exports CSV and JSON results, and passes its structural and model
smoke tests without modifying the user's existing workflow notebook or YAML
files.
