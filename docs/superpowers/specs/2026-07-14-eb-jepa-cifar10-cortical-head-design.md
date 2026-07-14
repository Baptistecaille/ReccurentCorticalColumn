# EB-JEPA CIFAR-10 versus cortical head

## Objective

Evaluate whether the recurrent cortical tree can replace the MLP projector in
the EB-JEPA CIFAR-10 image example while preserving the self-supervised test
objective within 2% and reducing compute on an NVIDIA A100.

This is a full-system efficiency comparison between two representation heads.
It is not a comparison between contextual JEPA predictors: the official
EB-JEPA image example is a JEA and contains no predictor.

## Source baseline

The baseline follows the official `facebookresearch/eb_jepa` image example:

- CIFAR-10 with two independently augmented views per image;
- CIFAR-adapted ResNet-18 backbone with a 512-dimensional output;
- a three-layer MLP projector with dimensions
  `512 -> 2048 -> 2048 -> 2048`;
- VICReg with standard-deviation coefficient 1 and covariance coefficient 80;
- LARS, 300 epochs, batch size 256, learning rate 0.3, weight decay `1e-4`;
- 10 warmup epochs followed by cosine learning-rate decay;
- BF16 automatic mixed precision;
- seeds 1, 1000, and 10000.

Primary references:

- <https://github.com/facebookresearch/eb_jepa>
- <https://github.com/facebookresearch/eb_jepa/blob/main/examples/image_jepa/README.md>
- <https://github.com/facebookresearch/eb_jepa/blob/main/examples/image_jepa/cfgs/default.yaml>

The implementation will preserve attribution and applicable Apache-2.0 notices
for any source adapted from EB-JEPA. A nested Git repository will not be
committed into this repository; only the minimal baseline components required
by the experiment will be integrated.

## Experimental systems

### EB-JEPA baseline

```text
augmented image
  -> CIFAR ResNet-18
  -> MLP projector (512 -> 2048 -> 2048 -> 2048)
  -> 2048-dimensional representation
```

### Cortical-head system

```text
augmented image
  -> identical CIFAR ResNet-18
  -> FixedTreePredictor (512 -> tree -> 2048)
  -> 2048-dimensional representation
```

The initial cortical configuration is:

- branching factor `n = 2`;
- maximum depth `L_max = 2`, giving seven columns;
- `dim_in = 512`;
- `dim_hidden = 256`;
- `dim_feedback = 128`;
- `dim_U = 256`;
- `dim_target = 2048`;
- lateral and feedback paths enabled.

These values are fixed before the main three-seed comparison. Any exploratory
tuning uses only the validation split and is reported separately.

## Data protocol

Use the standard CIFAR-10 training set without labels for self-supervised
training. Create one deterministic split shared by both systems:

- 45,000 images for training;
- 5,000 images for validation;
- the official 10,000-image test split, untouched until final evaluation.

Each training image produces two independent views using the official EB-JEPA
augmentation pipeline:

- random resized crop to 32 pixels with scale `[0.2, 1.0]`;
- color jitter with probability 0.8;
- grayscale with probability 0.2;
- solarization with probability 0.1;
- horizontal flip with probability 0.5;
- CIFAR-10 normalization.

The data ordering and random augmentation streams are paired between the two
systems for each seed. CIFAR-10 labels are never used during self-supervised
training or scoring.

## Loss

For two views `x1` and `x2`, the shared backbone and the selected head produce
`z1` and `z2` with shape `(batch, 2048)`. Both systems optimize the same VICReg
objective:

```text
L_total = L_invariance(z1, z2)
        + 1 * L_variance(z1, z2)
        + 80 * L_covariance(z1, z2)
```

The implementation must use the same loss class for both systems. It must log
the total, invariance, variance, and covariance terms separately. No linear
probe or supervised classification loss is created or optimized.

## Training protocol

Both systems use one shared trainer and differ only through `head.type`.

- optimizer: official EB-JEPA LARS configuration;
- epochs: 300;
- physical batch size: 256 on one A100;
- learning rate: 0.3;
- warmup: linear from `3e-5` for 10 epochs;
- post-warmup schedule: cosine decay to zero;
- weight decay: `1e-4`;
- momentum: 0.9;
- precision: BF16 autocast;
- seeds: 1, 1000, 10000;
- checkpoint: latest state plus periodic checkpoints every 50 epochs.

A checkpoint contains the model, optimizer, scheduler, scaler when applicable,
epoch, configuration, RNG states, and split indices. Resume must restore all of
them so that data and augmentation sequences remain reproducible.

## Validation and test scoring

Validation is used for health checks and any explicitly reported exploratory
tuning. The final test split is evaluated only after the configuration is
frozen.

Because VICReg requires paired views, test scoring uses five deterministic
augmentation pairs per image. The same pre-generated seeds and augmentation
pairs are used for both systems. The primary score is mean total VICReg test
loss over all pairs and images. The four loss components are reported as
mean and standard deviation across the three training seeds.

A run is invalid if representations collapse. Collapse checks include:

- non-finite loss or representation values;
- mean per-dimension standard deviation below 0.1;
- fewer than 90% of dimensions with standard deviation above 0.05.

These thresholds are health gates, not optimization targets.

## Compute benchmark

The reference hardware is one NVIDIA A100. Record the exact A100 memory variant,
CUDA version, PyTorch version, clock-management state, and software environment
with every benchmark result.

Measure both the representation head alone and the complete training step:

- trainable and total parameter counts;
- MACs and FLOPs for one batch;
- forward latency;
- forward-plus-backward latency;
- optimizer-step latency;
- images per second;
- peak allocated and reserved GPU memory.

Latency measurements exclude data loading and host-to-device transfer. Use
fixed device-resident synthetic tensors of the production shapes, 100 warmup
iterations, 500 measured iterations, CUDA events, and explicit synchronization
around the measured region. Both systems use the same BF16 mode and batch size.

## Success criterion

The cortical system succeeds when all three conditions hold:

1. none of its three runs triggers a collapse gate;
2. its mean total VICReg test loss is no more than 2% higher than the baseline;
3. its measured A100 compute is strictly lower for both the head-only benchmark
   and at least one complete-system metric among training-step latency, throughput,
   or peak memory.

Results must include raw per-seed values. A score improvement cannot compensate
for failed collapse checks, and a head-only compute reduction cannot be reported
as a full-system speedup unless the end-to-end measurement also improves.

## Component boundaries

The implementation will keep the following units independent:

- CIFAR-10 paired-view dataset and deterministic split management;
- official-compatible CIFAR ResNet-18;
- EB-JEPA MLP projector;
- adapter exposing `FixedTreePredictor` as a representation head;
- VICReg loss and collapse diagnostics;
- common trainer, checkpointing, validation, and test evaluation;
- A100 benchmark harness;
- configuration and result serialization.

Both heads implement one interface:

```python
def forward(features: torch.Tensor) -> torch.Tensor:
    """Map (batch, 512) features to (batch, 2048) representations."""
```

## Error handling

Configuration validation fails before training when dimensions are inconsistent,
the dataset split overlaps, the requested device is unavailable, or checkpoint
metadata does not match the active experiment. Training stops with an actionable
error on NaN, infinity, unexpected shapes, or invalid gradients. CUDA out-of-memory
errors report the active batch and model configuration without silently changing
the experimental protocol.

## Testing

Unit tests cover:

- deterministic, disjoint CIFAR-10 splits;
- paired-view output shapes and reproducibility;
- ResNet-18 and both head output shapes;
- the VICReg component values on collapsed and non-collapsed synthetic inputs;
- gradient propagation through every cortical column;
- checkpoint and RNG round-trips;
- compute-report schema and parameter counting.

Integration tests run one complete training step for each head using the same
synthetic batch and assert finite losses, matching output dimensions, and updates
only to the active model. A short CIFAR-10 smoke run verifies download/loading,
training, checkpoint resume, validation, and test evaluation before an A100 run
is launched.

## Deliverables

- reproducible baseline and cortical experiment configurations;
- a shared CIFAR-10 training and evaluation entrypoint;
- unit and integration tests;
- checkpoint/resume support;
- an A100 benchmark command;
- machine-readable JSON or CSV summaries containing scores and compute metrics;
- documentation with exact commands for smoke, training, test, and benchmark runs.
