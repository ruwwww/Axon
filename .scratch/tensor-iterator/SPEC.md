# Spec: Strided TensorIterator — make strides first-class

## Problem Statement

Every kernel in the CPU backend and every autograd backward pass assumes tensors are contiguous in memory. Strides are stored in `TensorType` but never read by any kernel. After a view operation (transpose, reshape, etc.), subsequent kernels read/write the wrong memory locations or force expensive materialization copies. This limits correctness to contiguous tensors only and will cause silent data corruption once view ops (slice, narrow, permute) are added.

## Solution

Introduce a `TensorIterator` abstraction in `include/axon/tensor/tensor_iterator.h` that correctly maps flat indices to memory offsets using a tensor's shape, strides, and storage offset. Migrate all element-wise CPU kernels and autograd backward passes to use `TensorIterator` instead of raw `data<T>()` with flat indexing. Add an `int64_t storage_offset_` field to Tensor (in elements) to enable zero-copy view operations.

## User Stories

1. As a framework developer, I want `TensorIterator` to compute correct memory offsets for any strided tensor, so that non-contiguous tensors produce correct results.
2. As a kernel author, I want `cpu::add` to work correctly on transposed inputs, so that view op outputs can be used in further computation without materialization.
3. As a kernel author, I want `cpu::relu` to work correctly on non-contiguous inputs, so that gradients flow through view ops correctly.
4. As an autograd developer, I want all backward passes to use TensorIterator for gradient accumulation, so that gradients from strided outputs accumulate correctly.
5. As a framework user, I want `transpose()` followed by any operation to produce numerically correct results, so that view ops compose safely.
6. As a framework user, I want `reshape()` followed by any operation to handle the resulting non-contiguous layout correctly.
7. As a developer adding new operations, I want to use the same strided access pattern as existing ops, so that my op works correctly with views by default.
8. As a test engineer, I want parameterized tests that run each kernel on contiguous and non-contiguous inputs, so that stride handling is verified for every op.

## Implementation Decisions

- **TensorIterator** lives in `include/axon/tensor/tensor_iterator.h`. It is constructed from a Tensor (reads shape, strides, dtype, and storage pointer + offset).
- **Interface**: `operator[]` (read-write, index-based) with an internal `is_contiguous()` fast-path branch that delegates to raw pointer access when the tensor is contiguous.
- **Storage offset**: a new `int64_t storage_offset_` field on Tensor, denominated in elements. `data<T>()` returns `(T*)storage_->data + storage_offset_`. View ops set this to a non-zero value.
- **Kernel migration is phased**: Phase 2 migrates element-wise kernels (add, sub, mul, div, relu, gelu, log_softmax, softmax). Phase 3 handles reduction kernels (matmul, conv2d, reduce_mean, batchnorm, layernorm).
- **Kernel signatures**: public API stays `cpu::add(Tensor& out, const Tensor& a, const Tensor& b)`. Internally the kernel constructs TensorIterators. An `*_iter` overload is extracted as a private detail for hot-path callers.
- **Backward passes** are migrated in Phase 4. Each switches from flat `data<float>()` + `numel()` loops to TensorIterator-based access.
- **No callback interface** in the first version. `operator[]` is the primary iteration mechanism.
- **No multi-index accessor** in the first version. Reduction kernels retain their current structure in Phase 2; multi-index support is added in Phase 3 if needed.
- **Operation registry**: the central switch in `Autograd::backward` is replaced with polymorphic dispatch (secondary priority — can be done independently).

## Testing Decisions

- Good tests verify behavior on both contiguous and explicitly non-contiguous tensors (transposed views, sliced views, strided views with offset).
- **TensorIterator**: new `tests/tensor/tensor_iterator_test.cpp` covering operator[] correctness, contiguous fast-path, edge cases (0-dim, 1-dim, scalars), and offset handling.
- **Backend kernels**: existing `tests/backend/cpu_backend_test.cpp` extended with non-contiguous input cases for each migrated element-wise kernel.
- **Autograd backward**: existing `tests/autograd/autograd_test.cpp` extended with strided-gradient backward tests for view ops (especially Transpose, Reshape) and gradient accumulation on non-contiguous tensors.

## Status

### Completed (sprint 7ce165b..4166ee4)

- [x] `storage_offset_` field added to Tensor, constructor updated, `data<T>()` uses offset (01)
- [x] TensorIterator class with `operator[]`, `is_contiguous()` fast-path, numel/ndim/shape/strides accessors (02)
- [x] Element-wise kernels (add, sub, mul, div, relu, gelu, log_softmax, softmax) use TensorIterator internally (03)
- [x] Backward passes use TensorIterator for gradient reads/writes (04, 06)
- [x] Reduction kernels (matmul, conv2d, reduce_mean, batchnorm, layernorm) use strided access (05)
- [x] Tests for contiguous and non-contiguous inputs for all migrated ops
- [x] Transpose and Reshape view operations propagate `storage_offset_` correctly

### Remaining

- [ ] Fix `const TensorIterator<float>` → `TensorIterator<const float>` in CPU backend for compile-time write protection on read-only inputs
- [ ] Remove dead variable `go_view_type` from autograd_test.cpp
- [ ] Remove scratch-work debugging comments from autograd_test.cpp
- [ ] Restore `std::fill` in reduce_mean (manual loop replaced idiomatic C++)
- [ ] Add `*_iter` overloads for hot-path callers (private detail extracted from each kernel)
- [ ] Migrate loss forward passes to TensorIterator (CrossEntropyLossOp, MSELossOp, L1LossOp)

## Out of Scope

- In-place dtype reinterpretation views (byte-level offset). Not needed without `tensor.view(dtype=...)` support.
- Multi-tensor / broadcasting iterator. Broadcasting is handled separately in the op's forward logic.
- Callback-based iteration (`forEach`). Can be added later as a convenience on top of `operator[]`.
- GPU backend changes.
- Higher-order gradients.
- MaxPool2d, AvgPool2d, Conv2D backward (still on raw pointers; not part of initial sprint)
- Polymorphic dispatch in Autograd::backward (deferred per spec; secondary priority)
