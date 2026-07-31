# 28 — Streamlined KernelKey (OpId, Device, DType, Provider) & Provider Abstraction

## What to build

Refactor `KernelRegistry` dispatch keys from string `"op_name:isa"` pairs to streamlined `KernelKey` structures containing `(OpId, Device, DType, Provider)` matching PyTorch ATen and oneDNN design, leaving ISA selection inside CPU provider execution.

## Acceptance criteria

- [ ] Define `enum class OpId : uint16_t` enum covering all core operators (`Add`, `Mul`, `MatMul`, `ReLU`, `GELU`, `Conv2D`, etc.)
- [ ] Implement `KernelKey` struct: `(OpId, Device, DType, Provider)`
- [ ] Refactor `KernelRegistry::register_kernel` and `dispatch` to use `KernelKey`
- [ ] All 193 unit test cases pass cleanly

## Status

backlog
