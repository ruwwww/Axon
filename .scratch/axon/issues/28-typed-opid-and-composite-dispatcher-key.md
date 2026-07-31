# 28 — Streamlined KernelKey (OpId, Device, DType, Provider) & Provider Abstraction

## What to build

Refactor `KernelRegistry` dispatch keys from string `"op_name:isa"` pairs to streamlined `KernelKey` structures containing `(OpId, Device, DType, Provider)` matching PyTorch ATen and oneDNN design, leaving ISA selection inside CPU provider execution.

## Acceptance criteria

- [x] Define `enum class OpId : uint16_t` enum covering all core operators (`Add`, `Mul`, `MatMul`, `ReLU`, `GELU`, `Conv2D`, etc.)
- [x] Implement `KernelKey` struct: `(OpId, Device, DType, Provider)`
- [x] Refactor `KernelRegistry::register_kernel` and `dispatch` to use `KernelKey`
- [x] All 195 unit test cases (2420 assertions) pass cleanly

## Status

completed
