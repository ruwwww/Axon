# 03 — Migrate element-wise kernels to TensorIterator

**What to build:** Port the element-wise CPU kernels (add, sub, mul, div, relu, gelu, log_softmax, softmax) to use TensorIterator internally. Each kernel constructs iterators from input/output Tensors and uses `operator[]` for element access. A private `*_iter` overload is extracted for potential hot-path use. Public signatures remain `Tensor&`. Extend backend tests with non-contiguous input cases for each kernel.

**Blocked by:** 02 — TensorIterator class

**Status:** completed

Closed by: `ab7e784` — Migrate element-wise CPU kernels to TensorIterator

- [ ] `cpu::add` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::sub` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::mul` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::div` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::relu` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::gelu` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::log_softmax` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `cpu::softmax` uses TensorIterator internally; works on non-contiguous inputs
- [ ] `tests/backend/cpu_backend_test.cpp` extended with non-contiguous test cases for each kernel
- [ ] All existing contiguous tests still pass
