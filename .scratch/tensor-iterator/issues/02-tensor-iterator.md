# 02 — TensorIterator class

**What to build:** Create `include/axon/tensor/tensor_iterator.h` with a `TensorIterator` class that reads shape, strides, dtype, and storage pointer (including offset) from a Tensor and provides `operator[]` for element access. Include an `is_contiguous()` check at construction with a fast-path branch inside `operator[]` that avoids stride arithmetic when contiguous. Add `numel()`, `ndim()`, `shape()`, `strides()` accessors.

**Blocked by:** 01 — Add storage_offset to Tensor

**Status:** completed

Closed by: `7791ffc` — Implement TensorIterator class

- [ ] `TensorIterator` constructible from `const Tensor&`
- [ ] `operator[]` returns correct element for contiguous tensors (fast path)
- [ ] `operator[]` returns correct element for transposed tensors (strided path)
- [ ] `operator[]` returns correct element for tensors with non-zero `storage_offset_`
- [ ] `is_contiguous()` returns true/false correctly
- [ ] New `tests/tensor/tensor_iterator_test.cpp` with tests for all access patterns
- [ ] Const-correct: `const TensorIterator` returns `const T&`
