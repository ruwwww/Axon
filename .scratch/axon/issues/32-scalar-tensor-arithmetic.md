# 32 — Scalar-Tensor Arithmetic & Missing Runtime Wiring

## What to build

Add scalar-tensor multiplication and division, and promote `sub` from backend-only to a full Runtime operation with autograd support.

These are general-purpose ops needed across many architectures (attention scaling, normalization, residual subtraction).

## Design note

Implement at the backend + autograd layer. The Runtime wiring is the current public API, but these ops will eventually migrate to a PyTorch-style `Tensor` method API (`tensor * scalar`, `a - b`). Keep the core logic in `cpu::` and autograd Nodes so the future API migration is just re-exposing, not reimplementing.

## Acceptance criteria

- [ ] `cpu::mul_scalar(Tensor& out, const Tensor& x, float scalar)` backend kernel
- [ ] `cpu::div_scalar(Tensor& out, const Tensor& x, float scalar)` backend kernel
- [ ] `runtime.mul_scalar(x, scalar)` and `runtime.div_scalar(x, scalar)` with autograd
- [ ] `runtime.sub(a, b)` promoted from backend to Runtime with autograd `SubNode`
- [ ] Unit tests for all new ops (forward + backward)
- [ ] All existing tests still pass

## Status

backlog
