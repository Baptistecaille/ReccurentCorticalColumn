# Recurrent Cortical Column — EB-JEPA CIFAR-10

This repository compares the baseline EB-JEPA projector with a recurrent
cortical-column predictor through a reproducible Train → Evaluate → Benchmark
→ Report workflow.

## Fresh-checkout setup

From the repository root, create a Python 3.12 environment and install the
project in editable mode with the notebook tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
python -m ipykernel install --user \
  --name recurrent-cortical-column \
  --display-name "Python (Recurrent Cortical Column)"
cd eb_jepa_cifar10_comparison
jupyter lab scripts/workflow.ipynb
```

In JupyterLab, select the **Python (Recurrent Cortical Column)** kernel. Run the
shared imports cell once, then edit only the parameter cell for the desired
section and run its execution cell. Launching JupyterLab from
`eb_jepa_cifar10_comparison` keeps the configuration and run paths relative to
the experiment project.

## Editable notebook parameters

| Section | Parameters | Meaning |
|---|---|---|
| Train | `TRAIN_CONFIG`, `TRAIN_OVERRIDES`, `TRAIN_SEED`, `TRAIN_OUTPUT_DIR`, `TRAIN_RESUME_FROM` | YAML configuration; OmegaConf dot-list overrides; training seed; run directory; optional checkpoint used to resume |
| Evaluate | `EVALUATE_CONFIG`, `EVALUATE_OVERRIDES`, `EVALUATE_CHECKPOINT`, `EVALUATE_PAIR_SEEDS`, `EVALUATE_OUTPUT` | YAML configuration; overrides; local best checkpoint; exactly five distinct pair seeds (default `11, 22, 33, 44, 55`); shared run artifact |
| Benchmark | `BENCHMARK_CONFIG`, `BENCHMARK_OVERRIDES`, `BENCHMARK_CHECKPOINT`, `BENCHMARK_OUTPUT` | YAML configuration; overrides; the same local checkpoint; the same shared run artifact |
| Report | `REPORT_RESULTS`, `REPORT_MARKDOWN_OUTPUT`, `REPORT_JSON_OUTPUT`, `REPORT_SCORE_TOLERANCE` | Exactly six complete artifact paths; distinct Markdown and JSON outputs; maximum relative score gap (default `0.02`, or 2%) |

An override is a string such as `"training.epochs=10"` in the relevant
`*_OVERRIDES` list. `TRAIN_RESUME_FROM=None` starts a new run; otherwise it must
point to the checkpoint to resume. Evaluate and Benchmark checkpoints must be
trusted local files: checkpoint loading uses `torch.load(weights_only=False)`,
which can execute code embedded in an untrusted pickle.

## Fixed experiment protocol

For each architecture (`baseline` and `cortical`), train with seeds `1`, `1000`
and `10000`. For every run, execute Evaluate and then Benchmark sequentially
against the same checkpoint and `result.json`; do not run them concurrently,
because both update that file. Evaluation requires exactly five distinct pair
seeds. Benchmark artifacts are accepted only for an A100 GPU (case-insensitive
name match), `bfloat16`, and batch size `256`; there is no CPU, H100, float32,
smaller-batch, or out-of-memory fallback.

Report consumes these six completed paths:

```text
runs/baseline/1/result.json
runs/baseline/1000/result.json
runs/baseline/10000/result.json
runs/cortical/1/result.json
runs/cortical/1000/result.json
runs/cortical/10000/result.json
```

The default `REPORT_SCORE_TOLERANCE=0.02` accepts a cortical VICReg score no
more than 2% above the baseline mean, in addition to the collapse and compute
gates recorded in both report outputs.
