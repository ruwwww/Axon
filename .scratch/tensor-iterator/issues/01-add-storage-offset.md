# 01 — Add storage_offset to Tensor

**What to build:** Add an `int64_t storage_offset_` field to Tensor (in elements of the tensor's dtype). Update `data<T>()` to return `(T*)storage_->data + storage_offset_`. Ensure TransposeOp::forward and ReshapeOp::forward correctly propagate offset (zero for allocator-created tensors, unchanged for views that share storage). Verify that existing view tests still pass and that `data<T>()` on a transposed view returns the same pointer as the original.

**Blocked by:** None — can start immediately.

**Status:** completed

Closed by: `cd8ddf0` — Add Tensor storage_offset support

- [ ] `storage_offset_` field added to `Tensor` class, constructor updated, `data<T>()` uses offset
- [ ] Allocator-created tensors have `offset_ == 0`
- [ ] TransposeOp::forward propagates `offset_` unchanged to output view
- [ ] ReshapeOp::forward propagates `offset_` unchanged to output view
- [ ] Existing view tests pass (transpose shares storage, reshape shares storage, data pointers match)
- [ ] Test: creating a tensor, transposing it, and verifying `data<T>()` returns correct address
