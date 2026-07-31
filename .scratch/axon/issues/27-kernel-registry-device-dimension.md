# 27 — KernelRegistry Device Dimension & Storage Accessor Seam

## What to build

Enhance `KernelRegistry` to support composite keys incorporating the `Device` dimension (`Device::CPU`, `Device::CUDA`, `Device::OpenCL`) alongside `ISA`, preparing Axon for multi-backend execution.

## Acceptance criteria

- [ ] `KernelKey` struct containing `(std::string op_name, Device device, ISA isa)`
- [ ] `KernelRegistry::register_kernel` and `lookup` accept `Device`
- [ ] Refactor `Tensor::data()` helper to encapsulate storage pointer access cleanly across backends
- [ ] All unit tests pass cleanly

## Status

backlog
