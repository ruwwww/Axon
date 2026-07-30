# 21 — Polymorphic Autograd Node & OpRegistry Dispatch

## What to build

Replace the 15-case `switch(node.op)` statement in `src/autograd/autograd.cpp` and the static `OpType` enum dependency with a polymorphic `OpRegistry` / `Node` dispatch system.

Each operation registers its backward evaluation handler (`Node` or `std::function<Expected<void>(Runtime&, const GraphNode&, GradientMap&)>`) into a central `OpRegistry` map. `Autograd::backward()` queries the registry to execute backward passes without needing a hardcoded switch statement or modifying `autograd.cpp` for every new op.

This satisfies the Open-Closed Principle and establishes the foundational seam required for multi-backend execution (`Ticket #22` AVX2 SIMD and `Ticket #23` OpenCL iGPU).

## Blocked by

None — can start immediately. Priority: P0 (Prerequisite for AVX2 and OpenCL backends).

## Acceptance criteria

- [ ] `OpRegistry` class introduced in `include/axon/autograd/op_registry.h`
- [ ] 15-case `switch(node.op)` in `Autograd::backward()` removed and replaced with dynamic `OpRegistry` lookup
- [ ] All 15 existing operations (`MatMul`, `ReLU`, `Add`, `Conv2D`, `GELU`, etc.) register their backward handlers in `OpRegistry`
- [ ] All 190 existing test cases in `axon_tests.exe` pass unchanged
- [ ] Adding a new operation can be done in its own file without editing `autograd.h` or `autograd.cpp`

## Status

ready-for-agent
