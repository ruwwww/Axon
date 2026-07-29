# 20 — Update stale README.md

## What to build

The current `README.md` describes an old C++17 API that no longer matches the codebase (references `tensor.hpp`, `layers.hpp`, `network.hpp`, `dataloader.hpp` as single-header files, and mentions build targets like `test_basic` and `cifar10_train` that don't exist).

Rewrite it to match the current architecture:

- Reflects C++20, CMake build, Catch2 tests
- Uses current namespace (`axon::`), domain terms (Runtime, Operation, Autograd, etc.)
- Provides accurate build/test instructions
- Points to examples/ for training scripts
- Remove stale references to non-existent files and targets

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] README accurately reflects the current API
- [ ] Build instructions produce a working build
- [ ] Test instructions run the full suite
- [ ] Example training command matches real example files

## Status

ready-for-agent
