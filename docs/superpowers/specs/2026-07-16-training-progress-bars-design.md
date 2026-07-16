# Training Progress Bars Design

## Goal

Make long training runs visibly progress in Colab and terminals without changing
the scientific protocol, numerical results, checkpoint behavior, or public
training API.

## User experience

`run` displays one persistent outer progress bar labelled `Epochs`. Each epoch
creates a transient inner progress bar labelled `Training` for the batches in
that epoch. The inner bar reports the running mean training loss. Once training
and validation finish, the outer bar reports the completed epoch's
`train_loss` and `val_loss`.

The implementation uses `tqdm.auto.tqdm`, which selects an appropriate notebook
or terminal renderer. The batch bar uses `leave=False` so completed epochs do
not fill the notebook with stale bars. The epoch bar remains visible and closes
normally when the run completes or raises.

## Code changes

- Import `tqdm` from `tqdm.auto` in `comparison.train`.
- Wrap the existing training-loader iteration in an inner progress bar.
- Update the inner postfix after each batch using the accumulated mean loss.
- Wrap the existing epoch range in an outer progress bar.
- Update the outer postfix after validation with the epoch's training and
  validation losses.

No configuration keys, function signatures, return values, optimizer steps,
scheduler steps, validation logic, or checkpoint rules change.

## Testing

Automated tests replace `comparison.train.tqdm` with a recording progress
double. A focused training-epoch test verifies the `Training` bar is created,
consumes every batch, and receives loss postfix updates. A focused `run` test
replaces its model, data, optimization, validation, and checkpoint collaborators
with small doubles, then verifies the `Epochs` bar and its train/validation
postfix without performing model computation.

The relevant focused tests run first, followed by the complete project test
suite. Tests must demonstrate the expected failure before production code is
changed.
