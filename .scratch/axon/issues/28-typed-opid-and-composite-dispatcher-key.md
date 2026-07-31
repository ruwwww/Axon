# 28 — Typed OpId Enum & Composite Dispatcher Key

## What to build

Refactor `KernelRegistry` dispatch keys from string `"op_name:isa"` pairs to typed composite `KernelKey` structures containing `(OpId, Device, DType, Layout, ISA, Provider)`, eliminating string parsing overhead and preventing typo bugs.

## Acceptance criteria

- [ ] Define `enum class OpId : uint16_t` enum covering all core operators (`Add`, `Mul`, `MatMul`, `ReLU`, `GELU`, `Conv2D`, etc.)
- [ ] Implement hashable `KernelKey` composite struct
- [ ] Refactor `KernelRegistry::register_kernel` and `dispatch` to use `KernelKey`
- [ ] All 193 unit test cases pass cleanly

## Status

backlog
