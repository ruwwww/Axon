# 13 — Sum reduction operation

## What to build

Sum operation that computes sum over specified dimensions.

- `SumOp::forward(tensor, dims, keepdim=false)` — computes sum along specified dimensions
- Backward: gradient is broadcast back to original shape (each element receives full gradient)
- CPU backend function `cpu::sum(out, input, dims)`
- Register in OpType enum and autograd dispatch
- Runtime::sum() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] Sum over single dim produces correct values
- [ ] Sum over all dims produces scalar
- [ ] keepdim preserves dimensions
- [ ] Backward gradient broadcast matches original shape
- [ ] Autograd integration test passes

## Status

ready-for-agent
