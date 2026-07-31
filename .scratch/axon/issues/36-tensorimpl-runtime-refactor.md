# 36 — TensorImpl Runtime Refactor & Stable Tensor Identity

## What to build

Refactor Axon's tensor object model to separate the lightweight frontend `Tensor` handle from the logical tensor implementation (`TensorImpl`), establishing stable identity across copies and enabling future runtime features.

This is an **architectural refactor** that should be completed **before** implementing tensor views, memory planning, graph compilation, or CUDA integration. It is not required for current eager execution milestones.

## Motivation

Current tensor identity is coupled to `Tensor` handle objects. Each `Tensor` instance carries a unique `TensorId` assigned at construction. When a `Tensor` is copied or moved, the new instance receives a different ID.

This causes architectural friction:

- Autograd `Node` must maintain parallel `input_ids_` vectors to preserve stable gradient-map lookups
- Future features (views, storage sharing, memory planning, compiler lowering) require a stable logical tensor object independent of frontend handle identity
- The current model makes it difficult to reason about "the same tensor" across API boundaries

See `docs/adr/0008-tensorimpl-and-stable-tensor-identity.md` for the full architectural decision record.

## Design

Introduce a `TensorImpl` layer:

```
Tensor (frontend value object / handle)
        │
        ▼
TensorImpl (logical tensor object)
        │
        ├── TensorId
        ├── Storage
        ├── Shape
        ├── Strides
        ├── DType
        ├── Device
        └── Autograd metadata
```

- `Tensor` becomes a lightweight handle holding a pointer/reference to a shared, immutable `TensorImpl`
- Copying a `Tensor` copies only the handle; multiple `Tensor` objects referring to the same logical tensor share the same `TensorImpl`
- `TensorId` lives in `TensorImpl`; logical tensor identity is stable across all copies of the handle
- `TensorType` continues to describe properties only (dtype, shape, layout, device); it does **not** contain identity information
- Autograd `Node` stores `std::vector<Tensor>` by value; `inputs_[i].id()` is always stable because copies share `TensorImpl`
- The current `input_ids_` workaround becomes unnecessary and is removed

## Benefits

- Stable logical identity across copies
- Cleaner autograd (no `input_ids_` plumbing)
- Storage sharing and tensor views
- Memory planner and buffer reuse pool compatibility
- Compiler integration and graph lowering
- Backend independence (CPU, CUDA, compiled kernels operate on the same logical tensor)
- Alignment with PyTorch (`c10::intrusive_ptr<TensorImpl>`) and TVM (`NDArray` handle referencing managed `Object`)

## Scope

- **Do not implement now.** This is a future architectural direction.
- **Do not modify current tensor semantics.**
- **Do not rewrite autograd.**
- Document the intended architecture before additional features increase migration cost.

## Priority

Medium–High.

Not required for current eager execution milestones, but should be completed **before** implementing:
- Tensor views or slice operations
- Memory planner or buffer reuse pool
- Graph compilation or lowering
- CUDA backend integration
- Any feature requiring stable logical tensor identity across handle copies

## Acceptance criteria

- [ ] `docs/adr/0008-tensorimpl-and-stable-tensor-identity.md` exists and is approved
- [ ] This ticket is linked from the ADR as the implementation vehicle
- [ ] Migration plan is drafted before implementation begins
- [ ] All existing tests continue to pass during any incremental migration

## Status

backlog
