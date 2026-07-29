# 11 — Slice operation

## What to build

Slice operation that extracts a sub-tensor sharing storage.

- `SliceOp::forward(tensor, dim, start, end, step=1)` — returns Tensor view sharing Storage with adjusted shape, strides, and data pointer
- Backward: gradient is scattered back into a zeros tensor at the sliced positions (zero gradient for non-sliced elements)
- Register in OpType enum and autograd dispatch

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Slice returns correct sub-shape and values
- [ ] Slice shares storage with parent (modifying one affects the other)
- [ ] Backward places gradient at correct positions, zeros elsewhere
- [ ] Autograd integration test passes

## Status

ready-for-agent
