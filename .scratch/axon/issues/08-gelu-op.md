# 08 — GELU activation operation

## What to build

GELU activation function with autograd support: `GELUOp` with `forward()` and `backward()`.

- Forward: `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- Backward: standard GELU derivative via the analytic formula
- CPU backend function `cpu::gelu(out, x)` with naive loop
- Register in OpType enum and Autograd::backward dispatch switch
- Runtime::gelu() method

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] GELU forward produces correct output for a known input
- [ ] GELU backward gradient matches finite differences
- [ ] Autograd integration test passes

## Status

ready-for-agent
