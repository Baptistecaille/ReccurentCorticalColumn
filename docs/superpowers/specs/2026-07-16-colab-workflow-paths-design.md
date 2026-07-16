# Colab workflow path correction

## Goal

Make `eb_jepa_cifar10_comparison/scripts/workflow.ipynb` use coherent,
Colab-only paths so training, evaluation, benchmarking, and reporting write to
the same visible project tree.

## Scope

- Target Google Colab only.
- Use `/content/ReccurentCorticalColumn/eb_jepa_cifar10_comparison` as the
  project root.
- Preserve `src` as the Python working directory so `comparison` imports keep
  working with the current notebook setup.
- Replace relative artifact paths and remaining macOS paths with paths derived
  from the fixed Colab project root.
- Add a small diagnostic cell that displays the working directory and generated
  run files.

## Notebook design

The setup section defines `PROJECT_ROOT`, `SOURCE_ROOT`, `DATA_ROOT`,
`RUNS_ROOT`, and `REPORTS_ROOT` as `pathlib.Path` values after cloning. Dataset
download and extraction target `DATA_ROOT`. The shared-import cell changes to
`SOURCE_ROOT` before importing the local `comparison` package.

Each workflow parameter cell derives its configuration, checkpoint, output,
and dataset paths from those shared roots. Train writes checkpoints below
`RUNS_ROOT`; Evaluate and Benchmark update the same absolute `result.json`;
Report consumes those six absolute result paths and writes below
`REPORTS_ROOT`.

The diagnostic cell prints `Path.cwd()` and recursively lists files below
`RUNS_ROOT`, making the actual output location explicit in Colab.

## Error handling and validation

Existing training and workflow errors remain unchanged. The notebook will be
validated structurally with `nbformat`. A focused automated check will assert
that executable workflow cells contain no `/Users/` paths, that all artifact
paths derive from the shared roots, and that Train, Evaluate, and Benchmark
target the same run directory for seed 1.

Full top-to-bottom execution is not part of local validation because cloning,
CIFAR-10 download, training, and the strict A100 benchmark require a Colab A100
runtime.
