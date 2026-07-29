# 12 — Mean reduction operation

## What to build

Mean operation that computes the mean over specified dimensions.

- `MeanOp::forward(tensor, dims, keepdim=false)` — computes mean along specified dimensions
- Backward: gradient is broadcast back to original shape, divided by reduction size
- CPU backend function `cpu::mean(out, input, dims)`
- Register in OpType enum and autograd dispatch
- Runtime::mean() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Mean over single dim produces correct values
- [ ] Mean over multiple dims produces correct values
- [ ] keepdim preserves dimensions
- [ ] Backward gradient broadcast matches original shape
- [ ] Autograd integration test passes

## Status

ready-for-agent
