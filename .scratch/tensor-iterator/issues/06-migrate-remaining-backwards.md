# 06 — Migrate remaining backward passes to TensorIterator

**What to build:** Switch the backward passes for reduction/view operations (Conv2DOp, MaxPool2dOp, AvgPool2dOp, BatchNormOp, LayerNormOp, MeanOp, ReshapeOp, TransposeOp) to use TensorIterator for gradient access and accumulation. TransposeOp::backward currently manually reconstructs multi-dimensional indices — replace with a TensorIterator that reads from the strided grad_out and writes to a contiguous grad. ReshapeOp::backward currently does a memcpy — replace with TensorIterator read/write.

**Blocked by:** 05 — Migrate reduction kernels to TensorIterator

**Status:** completed

Closed by: `93f53aa` — Migrate remaining backward passes to TensorIterator

- [ ] Conv2DOp::backward uses TensorIterator for gradient accumulation
- [ ] MaxPool2dOp::backward uses TensorIterator for gradient scatter
- [ ] AvgPool2dOp::backward uses TensorIterator for gradient scatter
- [ ] BatchNormOp::backward uses TensorIterator for gradient reads/writes
- [ ] LayerNormOp::backward uses TensorIterator for gradient reads/writes
- [ ] MeanOp::backward uses TensorIterator for gradient broadcast
- [ ] ReshapeOp::backward uses TensorIterator instead of memcpy
- [ ] TransposeOp::backward uses TensorIterator instead of flat-index reconstruction
- [ ] Tests: strided-gradient backward for each op
- [ ] All existing autograd tests still pass
