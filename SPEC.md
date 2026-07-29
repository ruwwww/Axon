# Axon v1 — Phase 1 Spec

## Problem Statement

Building a deep learning framework from scratch to understand how modern AI runtimes work. Current state: a high-level architecture spec exists, but no code. The goal is an executable framework that can train neural networks on CIFAR-10 and MNIST with CPU execution, eager autograd, and GGML-style quantized storage.

## Solution

A minimal, layered C++20 framework with six subsystems: Storage, Tensor, Backend (CPU), Runtime, Autograd, and Neural Network API (Module/Parameter/Optimizer). Each layer has a narrow interface; the Runtime is the public execution entry point. All cross-subsystem error propagation uses `Expected<T>` return types.

## User Stories

1. As a framework user, I want to create tensors with `Runtime::zeros()`, `Runtime::randn()`, `Runtime::ones()`, so that I can prepare input data without managing memory.
2. As a framework user, I want to perform operations like `Runtime::matmul()`, `Runtime::relu()`, `Runtime::conv2d()`, so that I can build computation graphs with autograd recording.
3. As a framework user, I want to define model parameters explicitly via `Module::register_parameter()`, so that the framework knows which tensors are trainable.
4. As a framework user, I want to define neural network layers (Linear, Conv2D, BatchNorm, LayerNorm, Embedding, Dropout, Flatten, Sequential, Residual) as Modules, so that I can compose architectures.
5. As a framework user, I want `loss.backward()` to compute all parameter gradients via reverse-mode autograd, so that I can train models.
6. As a framework user, I want optimizers (SGD, Adam, AdamW) accepting `vector<Parameter*>` and mutating them in-place, so that training loops are straightforward.
7. As a framework user, I want a synchronous DataLoader that collates batches from a Dataset, so that I can iterate training data without thread complexity.
8. As a framework user, I want losses as Operations (CrossEntropy, MSE, L1, NLL) with fused analytic backward, so they participate in autograd like any other op.
9. As a framework user, I want native binary serialization (`.axon` format) for saving/loading tensors and checkpoints, so that I can persist trained models.
10. As a framework engineer, I want a clean backend seam (namespace of free functions in `namespace cpu`), so that a future GPU backend can be added without changing Tensor's interface.
11. As a framework engineer, I want the Graph to be a linear node list with no optimization or scheduling, so that autograd is simple and debuggable.
12. As a framework engineer, I want `Expected<T>` error returns across subsystem boundaries, so that failures are explicit and composable.
13. As a framework engineer, I want to write tests bottom-up: Backend unit → Tensor unit → Operation unit → Autograd integration → Optimizer unit → Serialization roundtrip.

## Implementation Decisions

### Architecture

- **Six subsystems**: Storage, Tensor, Backend (CPU), Runtime, Autograd, NN API (Module/Parameter/Optimizer/Loss/Dataset/DataLoader).
- **Tensor** holds `std::shared_ptr<Storage>` for simplicity in Phase 1. The original spec's non-owning pointer can be restored later.
- **TensorType** is immutable: shape, stride, dtype, device, quantization.
- **Storage** is reference-counted via `shared_ptr`. Owns a raw byte buffer, size, alignment, and an optional `QuantizationDescriptor`.
- **QuantizationDescriptor** describes block size, packing, scales, and decode rule. Belongs to Storage, not Tensor.

### Runtime and Execution

- **Runtime** is the public API. Users call `runtime.matmul(a, b)` which delegates to `MatMulOp::forward()`.
- **Runtime** owns the Allocator and the Graph reference.
- **Allocator** creates Storage blocks. Not a global singleton — owned by Runtime.
- **Operations** are stateless functors implementing `forward()` and `backward()`.
- **Operations own graph recording**: `forward()` allocates outputs, calls the CPU backend kernel, then (if `requires_grad`) appends a `GraphNode`.
- **GraphNode** stores: operation type, input tensors, output tensor, and a non-owning `Runtime*` (stored during forward, used by backward for allocation).
- **Backend** is a namespace `cpu::` of free functions. It receives `Tensor*` pointers. Never accesses the Graph.

### Autograd

- **Autograd** owns the Graph and GradientMap.
- **GradientMap** maps TensorId → gradient Tensor during backward. Accumulates in-place.
- **Backward traversal**: create initial gradient, iterate nodes in reverse, call `op->backward(node, grads)`, accumulate. After traversal, copy gradients to `Tensor::grad`.
- **Losses** are Operations with fused analytic backward (e.g., log-softmax + NLL fused in CrossEntropy).

### Modules and Parameters

- **Parameter** wraps a Tensor with a separate gradient Tensor and a trainable flag.
- **Module** provides `forward()`, `parameters()`, `train()`, `eval()`. Parameters are registered manually via `register_parameter(name, &param)`.
- **Optimizer** receives `Runtime&` and `vector<Parameter*>`. Owns state tensors (momentum buffers, Adam moments). Eager allocation on construction. `step()` mutates in-place. `zero_grad()` zeros each parameter's gradient.

### Data

- **Dataset** interface: `size()`, `get(index)`. Phase 1: pre-downloaded files only.
- **DataLoader**: synchronous, single-threaded, no prefetch. Iterates indices, calls `dataset.get(i)`, collates into batch Tensor.

### Serialization

- Free functions, not methods on Storage or Module.
- `.axon` format: magic header, version, tensor count, then per-tensor (name, dtype, quant_type, shape, raw bytes).
- Checkpoints iterate `module.parameters()`, write each `(name, tensor)` pair. Loading matches names to registered parameters.
- GGUF weights: Phase 1 targets GGML-inspired packing *strategy* (block layout for Q8_0 etc.) but does not parse the GGUF container format.

### Error Handling

- All cross-subsystem public APIs return `Expected<T>` or `Status`.
- Exceptions allowed only within a subsystem for unrecoverable internal bugs.

### Build and Test

- **Build**: CMake 3.22+, GCC 12 or Clang 14+, `-std=c++20`.
- **Test framework**: Catch2 v3.
- **Test order (bottom-up)**: Backend unit → Tensor unit → Operation unit → Autograd integration → Optimizer unit → Serialization roundtrip.

## Testing Decisions

### What makes a good test

- Test external behavior, not implementation details.
- Each test asserts one logical thing.
- Prefer known numerical values over comparing to a second implementation.
- Backend tests verify kernel math directly with raw arrays, not Tensors.
- Tensor tests verify views share Storage correctly and refcounting works.
- Operation tests verify the right backend function is called with the right shapes — do not re-prove kernel math.
- Autograd tests verify finite differences match on small graphs.
- Optimizer tests verify parameter update matches hand-calculated formula.
- Serialization tests verify save/load produces identical parameters.

### Modules to test

- `cpu` backend kernels (add, matmul, relu, softmax, layernorm, conv2d)
- Tensor (creation, views, stride, layout, refcounting, dtype conversion)
- Each Operation (MatMulOp, ReLUOp, CrossEntropyOp, etc.)
- Autograd (full forward+backward on a small graph)
- Optimizer (SGD, Adam on a single Parameter with known gradient)
- Serialization (save/load roundtrip on a Module)

### What NOT to test

- DataLoader internals beyond `get(i)` returning the right shape.
- Per-op backward in isolation (proven by autograd integration).
- Graph traversal mechanics (proven by autograd integration).

## Out of Scope

- CUDA or any GPU backend.
- Distributed training.
- JIT compilation or graph optimization.
- Compiler IR or dynamic shape optimization.
- ONNX export or import.
- Async DataLoader workers (`num_workers > 0`).
- Built-in dataset download or caching.
- GGUF container format parsing.
- Thread safety or multi-threaded execution.

## Further Notes

- No global state, no singletons, no macros.
- RAII everywhere. Raw pointers are non-owning only.
- Memory ownership: Tensor (`shared_ptr<Storage>`), Graph (owns nodes), Module (owns parameters), Optimizer (owns state tensors).
- Milestones: M1 (Tensor, Storage, MatMul, ReLU, Autograd) → M2 (Linear, SGD, MNIST) → M3 (Conv2D, AdamW, CIFAR10) → M4 (ResNet18, CIFAR10) → M5 (GGML quantized inference/training research).
