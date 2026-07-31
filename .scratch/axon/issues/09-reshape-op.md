# 09 — Reshape operation

## What to build

Reshape/view operation that changes tensor shape without copying storage.

- `ReshapeOp::forward(tensor, new_shape)` — validates new shape has same numel, returns Tensor with same Storage and updated TensorMetadata (contiguous strides)
- Backward: gradient is reshaped back to original shape (identity with shape change)
- Only contiguous tensors in Phase 1 (non-contiguous fails gracefully)
- Register in OpType enum and autograd dispatch

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Reshape changes shape without copying storage
- [ ] Reshape validates total element count matches
- [ ] Backward returns gradient reshaped to original shape
- [ ] Autograd integration test passes
- [ ] Error on invalid shape returns Expected error

## Status

ready-for-agent
