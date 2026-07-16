# Phase 14 Interactive Workflow Notebook Design

## Goal

Replace the four planned command-line Python scripts with one interactive
notebook at `eb_jepa_cifar10_comparison/scripts/workflow.ipynb`. The notebook
must expose training, evaluation, benchmarking, and report generation without
duplicating scientific logic outside `src/comparison`.

## Intentional Roadmap Deviation

Phase 14 originally specifies four files with `argparse`, four `main()`
functions, `if __name__ == "__main__"` guards, and `--help` checks. The approved
design replaces those CLI-specific requirements with one interactive Jupyter
workflow. It therefore does not provide `argparse`, `main()` functions, or
direct `python scripts/<name>.py` invocation.

The scientific protocol and the four distinct entry points remain unchanged.

## Notebook Structure

The notebook contains an introductory Markdown cell, one shared imports cell,
and four independently runnable sections in this order:

1. **Train**
   - Parameters: configuration path, training seed, output directory, optional
     resume checkpoint, and dotted configuration overrides.
   - Execution: load the configuration and invoke the training workflow once.
   - Output: display the produced best-checkpoint path.

2. **Evaluate**
   - Parameters: configuration path, checkpoint path, exactly five pair seeds,
     training seed, output JSON path, and optional configuration overrides.
   - Execution: invoke one package-level evaluation-to-JSON workflow.
   - Output: display the evaluation JSON path.

3. **Benchmark**
   - Parameters: configuration path, checkpoint path, output JSON path, and
     optional configuration overrides.
   - Execution: invoke one package-level checkpoint benchmark workflow.
   - Output: display the benchmark JSON path.

4. **Report**
   - Parameters: three baseline result JSON paths, three cortical result JSON
     paths, Markdown output path, JSON output path, and score tolerance.
   - Execution: invoke one package-level report workflow.
   - Output: display both report paths and the final comparison decision.

Each section has a Markdown explanation, one clearly labeled editable
parameter cell, one execution cell, and one concise result display. A section
can run independently after the shared imports cell; running an earlier
scientific section is not a prerequisite when its input files already exist.

## Package Boundaries

The notebook only translates editable Python variables into calls to public
functions under `src/comparison`. It must not contain:

- model or optimizer construction;
- checkpoint deserialization;
- CIFAR-10 dataset construction;
- training or evaluation loops;
- CUDA timing or memory logic;
- scientific aggregation or comparison formulas;
- JSON schema reconstruction.

Where the existing modules expose primitives rather than a complete file-based
workflow, focused orchestration functions will be added under
`src/comparison`. These functions will compose the already tested Phase 1–13
functions, write the requested artifacts, and return paths or immutable result
objects suitable for notebook display.

The train section may directly compose `load_config` and `comparison.train.run`
because `run` already owns the full scientific workflow. Evaluation,
benchmarking, and reporting require package-level file-oriented orchestration
so the notebook does not reconstruct models or dataclasses.

## Data Contracts

Evaluation and benchmark artifacts must contain enough shared metadata to form
a `SeedResult`: normalized head type, training seed, checkpoint hash,
evaluation data, benchmark data, GPU name, dtype, and batch size. Report input
loading validates these fields, the required seed set `{1, 1000, 10000}`, and
architecture consistency before calling the existing Phase 13 aggregation and
decision functions.

All generated JSON uses UTF-8, stable indentation, sorted keys, and
`allow_nan=False`. Parent directories are created by package functions, not by
notebook cells.

## Error Handling

Errors remain visible as normal notebook tracebacks. Package functions provide
the meaningful validation messages. The notebook does not catch out-of-memory
errors, retry with altered parameters, choose another architecture, or silently
fall back from CUDA. Architecture always comes from the loaded configuration.

## Documentation

`eb_jepa_cifar10_comparison/README.md` will document:

- how to open `scripts/workflow.ipynb`;
- the four sections and their editable parameters;
- the normal train → evaluate → benchmark → report flow;
- how to execute one section independently from existing artifacts;
- the A100 requirement for the benchmark section.

## Verification

Phase 14 verification replaces the original four `--help` checks with checks
appropriate to an interactive notebook:

- the notebook is valid `nbformat` JSON and has the expected four sections;
- each section includes a parameter cell and an execution cell;
- importing the shared cell does not download CIFAR-10 or initialize CUDA;
- notebook execution calls are tested with package functions replaced by
  harmless fakes, proving parameter forwarding without scientific execution;
- no scientific implementation appears in notebook source cells;
- existing Phase 1–13 regression checks still pass;
- every Python module compiles and `git diff --check` reports no errors.

## Completion Criteria

- `scripts/workflow.ipynb` contains all four approved interactive sections.
- Each section delegates scientific work to `src/comparison`.
- Each produced artifact path is displayed.
- No training, evaluation, benchmark, or reporting science is duplicated.
- Notebook usage and invocation order are documented.
- Notebook-specific checks and all prior regression checks pass.
