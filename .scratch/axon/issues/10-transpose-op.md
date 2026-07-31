# 10 — Transpose operation

## What to build

Transpose operation that swaps tensor dimensions.

- `TransposeOp::forward(tensor, dim1, dim2)` — returns Tensor with same Storage, swapped strides in TensorMetadata
- Backward: transpose same two dims (self-inverse)
- Register in OpType enum and autograd dispatch
- Runtime::transpose() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Transpose swaps dimensions correctly (shape and strides)
- [ ] Transpose does not copy storage
- [ ] Backward returns gradient with same axes swapped
- [ ] Autograd integration test passes
- [ ] Error on invalid dim returns Expected error

## Status

ready-for-agent
