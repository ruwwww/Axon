# 14 — Max reduction operation

## What to build

Max operation that computes maximum over specified dimensions.

- `MaxOp::forward(tensor, dim, keepdim=false)` — computes max along one dim, returns (values, indices)
- Backward: gradient flows only to the max elements (scatter via indices), zero elsewhere
- CPU backend function `cpu::max(out, indices, input, dim)`
- Register in OpType enum and autograd dispatch
- Runtime::max() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Max over a dim returns correct values and indices
- [ ] keepdim preserves dimensions
- [ ] Backward only passes gradient to max elements, zeros elsewhere
- [ ] Autograd integration test passes

## Status

ready-for-agent
