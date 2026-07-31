# 27 — KernelRegistry Device Dimension & Storage Accessor Seam

## What to build

Enhance `KernelRegistry` to support composite keys incorporating the `Device` dimension (`Device::CPU`, `Device::CUDA`, `Device::OpenCL`) alongside `ISA`, preparing Axon for multi-backend execution.

## Acceptance criteria

- [x] `KernelKey` struct containing `(OpId, Device, DType, Provider)` — superseded by Ticket #28
- [x] `KernelRegistry::register_kernel` and `lookup` accept `Device` via `KernelKey`
- [ ] Refactor `Tensor::data()` helper to encapsulate storage pointer access cleanly across backends (deferred — separate concern)
- [x] All 195 unit test cases (2420 assertions) pass cleanly

## Status

completed (Device dimension solved via Ticket #28 KernelKey; Tensor::data() accessor refactor deferred)
