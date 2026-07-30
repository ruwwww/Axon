# 04 — Migrate backward passes for element-wise ops to TensorIterator

**What to build:** Switch the backward passes for element-wise operations (MatMulOp, ReLUOp, AddOp, GELUOp, CrossEntropyLossOp, MSELossOp, L1LossOp) from flat `data<float>()` + `numel()` indexing to TensorIterator-based gradient reads/writes. Gradient accumulation via `cpu::add` already works on non-contiguous tensors after ticket 03. Verify strided gradient correctness for view op compositions (e.g., transpose → matmul → backward).

**Blocked by:** 03 — Migrate element-wise kernels to TensorIterator

**Status:** completed

Closed by: `21e7e49`, `93f53aa` — Migrate element-wise/remaining backward passes

- [ ] MatMulOp::backward constructs iterators; gradient reads/writes use strided access
- [ ] ReLUOp::backward uses TensorIterator for input reads and gradient writes
- [ ] AddOp::backward uses TensorIterator for gradient reads and bias-sum flows
- [ ] GELUOp::backward uses TensorIterator for input reads and gradient writes
- [ ] CrossEntropyLossOp::backward uses TensorIterator for log_softmax reads
- [ ] MSELossOp::backward uses TensorIterator for gradient reads/writes
- [ ] L1LossOp::backward uses TensorIterator for gradient reads/writes
- [ ] Tests: strided-gradient backward for transpose → matmul, reshape → relu, and similar view-op compositions
- [ ] All existing autograd tests still pass
