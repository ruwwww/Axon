# 05 — Migrate reduction kernels to TensorIterator

**What to build:** Port the reduction/multi-dimensional CPU kernels (matmul, conv2d, reduce_mean, batchnorm, layernorm) to use TensorIterator for their element-access patterns where applicable. These kernels have nested loops and complex multi-dimensional indexing, so the migration is less mechanical than element-wise ops. Add multi-index accessor support to TensorIterator (`at(indices)` or `offset(indices)`) if needed.

**Blocked by:** 02 — TensorIterator class

**Status:** completed

Closed by: `4166ee4` — Migrate reduction kernels to TensorIterator

- [ ] `cpu::matmul` uses TensorIterator or multi-index access; works on non-contiguous A or B
- [ ] `cpu::conv2d` uses strided access; works on non-contiguous input/weight
- [ ] `cpu::reduce_mean` uses TensorIterator; works on non-contiguous inputs
- [ ] `cpu::batchnorm` uses TensorIterator; works on non-contiguous inputs
- [ ] `cpu::layernorm` uses TensorIterator; works on non-contiguous inputs
- [ ] `tests/backend/cpu_backend_test.cpp` extended with non-contiguous test cases for each
- [ ] All existing reduction tests still pass
