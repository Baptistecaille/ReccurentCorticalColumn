# MLP Encoder Design

## Goal

Reduce CIFAR-10 training time by replacing the ResNet-18 encoder used by the
default Baseline and Cortical experiments with a lightweight MLP encoder. This
change isolates encoder compute first: the two heads, the VICReg objective, and
its 2,048-dimensional projection space remain unchanged.

If training remains too slow after this change, reducing the VICReg projection
dimension to 1,024 and then 512 will be considered separately.

## Architecture

Add an `MLPEncoder` that accepts normalized CIFAR-10 batches shaped
`(batch, 3, 32, 32)` and applies:

1. `Flatten`, producing 3,072 input values per image;
2. `Linear(3072, 512)`;
3. `LayerNorm(512)`;
4. `GELU`.

The encoder returns `(batch, 512)`, preserving the existing contract between
the backbone and both representation heads. Its trainable parameter count is
1,574,400: 1,573,376 for the linear layer and 1,024 for LayerNorm.

`ResNet18` remains available. A backbone factory selects either `MLPEncoder`
or `ResNet18` from `model.backbone`. The default Baseline and Cortical YAML
configurations select `mlp`, ensuring that both sides of the comparison still
share exactly the same encoder.

## Configuration and validation

`model.backbone` accepts exactly `mlp` or `resnet18`, case-insensitively. An
unknown value fails during configuration validation and during model
construction with a message that lists the accepted values.

`model.feature_dim` remains fixed at 512. `cortical.dim_in` must continue to
equal `model.feature_dim`. `model.output_dim` and `cortical.dim_target` remain
2,048 in this change.

## Data flow

The training and evaluation flow remains:

`CIFAR-10 image -> shared encoder -> selected head -> VICReg projection`

Only the shared encoder implementation changes in the default configurations.
Augmentations, paired views, loss computation, optimizer, scheduler,
checkpointing, evaluation, benchmarking, and reporting retain their current
interfaces and behavior.

## Checkpoint compatibility

New MLP runs write their resolved configuration into each checkpoint as before.
An MLP checkpoint can only be loaded with a matching MLP configuration. Existing
ResNet-18 checkpoints remain loadable with their original ResNet-18
configuration; loading them under an MLP configuration must fail strictly
rather than partially loading incompatible weights.

## Testing

Automated tests cover:

- `MLPEncoder` maps `(B, 3, 32, 32)` to `(B, 512)`;
- its trainable parameter count is 1,574,400;
- model construction selects the requested MLP or ResNet-18 backbone;
- both default experiment configurations select the MLP and build outputs with
  unchanged feature and projection dimensions;
- configuration validation accepts `mlp` and rejects unknown backbone names;
- existing device, training, evaluation, checkpoint, benchmark, and workflow
  tests continue to pass.

## Success criteria

The change is complete when both default configurations train through the same
public workflow with `MLPEncoder`, preserve `(B, 512)` features and `(B, 2048)`
projections, and the full test suite passes. A small local timing comparison may
be reported as an observation, but no machine-specific speed threshold is a
correctness requirement.
