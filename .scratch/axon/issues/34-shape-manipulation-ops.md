# 34 — Shape Manipulation Ops (cat, split, view)

## What to build

Add tensor shape manipulation operations needed for multi-head attention and general-purpose tensor programming: concatenation, splitting, and view/contiguous reshaping.

## Design note

These are foundational tensor ops, not transformer-specific. Implement at backend + autograd layer. `cat` and `split` are inverses and their backward implementations mirror each other. `view` is a zero-copy reshape when contiguous, otherwise requires a copy.

The Runtime wiring is the current API but will eventually migrate to `Tensor` methods (`tensor.view(shape)`, `Tensor::cat(tensors, dim)`, `tensor.split(size, dim)`).

## Acceptance criteria

- [ ] `runtime.cat(tensors, dim)` — concatenate tensors along a dimension, with `CatNode` autograd (backward splits grad)
- [ ] `runtime.split(tensor, split_size, dim)` — split tensor into chunks, with `SplitNode` autograd (backward cats grads)
- [ ] `runtime.view(tensor, shape)` — zero-copy reshape when contiguous, copy otherwise
- [ ] Multi-head attention reshape verified: `(B, T, d_model)` → `(B, T, n_heads, d_k)` → transpose → `(B, n_heads, T, d_k)`
- [ ] Unit tests for all ops (forward + backward)
- [ ] All existing tests still pass

## Status

backlog
