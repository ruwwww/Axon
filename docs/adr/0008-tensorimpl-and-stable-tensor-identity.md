# ADR-0008: TensorImpl & Stable Tensor Identity (Architectural Direction)

## Status

Accepted — Implemented

## Context

Axon's current tensor model couples logical tensor identity to individual `Tensor` handle objects. Each `Tensor` instance carries a unique `TensorId` assigned at construction. When a `Tensor` is copied or moved, the new instance receives a different ID.

This design was sufficient for early eager execution, but it introduces architectural friction as the runtime matures:

- Autograd `Node` subclasses must maintain a parallel `input_ids_` vector to preserve stable gradient-map lookups, since `inputs_[i].id()` may refer to a moved-from or copied tensor with a different ID.
- Future features — tensor views, storage sharing, copy-on-write, memory planning, graph lowering — require a stable logical tensor object independent of frontend handle identity.
- The current model makes it difficult to reason about "the same tensor" across API boundaries, which is a prerequisite for compiler integration and backend independence.

This ADR documents the intended long-term architecture. No implementation changes are proposed at this time.

## Decision

Future versions of Axon will introduce a `TensorImpl` layer that separates the lightweight frontend handle from the logical tensor object:

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

- **`Tensor` becomes a lightweight handle**: It holds a pointer/reference to a shared, immutable `TensorImpl`. Copying a `Tensor` copies only the handle. Multiple `Tensor` objects referring to the same logical tensor share the same `TensorImpl`.
- **`TensorId` lives in `TensorImpl`**: Logical tensor identity is stable across all copies of the handle. `tensor.id()` returns `impl->id()`.
- **`TensorType` continues to describe properties only**: `TensorType` describes dtype, shape, layout, and device. It does **not** contain identity information. Identity is a property of the logical tensor object, not its type.
- **Autograd `Node` stores `std::vector<Tensor>` by value**: Because all copies share the same `TensorImpl`, `inputs_[i].id()` is always stable. The current `input_ids_` workaround becomes unnecessary and is removed.
- **Backward propagation uses stable IDs**: `GradientMap` lookups in `Node::apply()` use `inputs_[i].id()` directly. No silent failures from identity mismatch.

## Explicitly Rejected

**Storing `TensorId` inside `TensorType`.**

`TensorType` is a descriptor of a tensor's shape, layout, dtype, and device. Two tensors can have the same type (e.g., `Float32[3, 4]` on CPU) but be completely different logical tensors with different identities. Identity is orthogonal to type and must not be encoded in `TensorType`.

## Architectural Goals

`TensorImpl` is the stable runtime object between frontend APIs and execution backends. It prepares Axon for:

- Storage sharing (multiple logical tensors aliasing the same underlying memory)
- Tensor views (sliced, strided, or broadcasted tensors sharing storage with a parent)
- Copy-on-write semantics
- Memory planner and buffer reuse pool
- Storage planner
- CUDA unified tensor representation
- Compiler integration and graph lowering
- Backend independence (CPU, CUDA, compiled kernels operating on the same logical tensor)

The data flow remains:

```
Tensor API
     │
Runtime
     │
Autograd
     │
Kernel Registry
     │
CPU / CUDA / Compiler
```

Each subsystem manipulates `Tensor` handles that reference shared `TensorImpl` objects. No subsystem needs to manage logical identity separately.

## Relationship with Existing Architecture

This change does **not** replace:
- `Runtime`
- `KernelRegistry` / `KernelKey`
- `Dispatcher`
- `Allocator`
- `Autograd` / `Graph` / `GradientMap`

It refactors the `Tensor` object model that all of the above depend on. After the transition:

- `Runtime` methods accept and return `Tensor` handles as before.
- `Allocator` creates `Storage` and returns a `Tensor` handle wrapping a new `TensorImpl`.
- `Autograd::Node` stores `std::vector<Tensor>` inputs by value; identity is preserved because copies share `TensorImpl`.
- Backend kernels receive `Tensor*` or `Tensor&` as they do today; the underlying `TensorImpl` provides stable access to storage and metadata.

## Comparison with Existing Frameworks

### PyTorch

PyTorch's `Tensor` is a lightweight handle (`c10::intrusive_ptr<Impl>`). The `TensorImpl` object owns storage, metadata, version counters, and runtime state. Copying a `Tensor` increments the `intrusive_ptr` refcount; all copies share the same `TensorImpl` and thus the same logical identity. Autograd `Node` stores `std::vector<Tensor>` inputs and uses the `Impl`'s version-stable ID for gradient map lookups.

This is the primary architectural inspiration for Axon's `TensorImpl`.

### GGML

GGML uses long-lived graph tensor objects owned by a `ggml_context`. Graph nodes hold non-owning raw pointers to canonical tensor objects. This works well for static execution but is less representative of Axon's dynamic eager runtime where tensors are created and consumed across API boundaries without a single owning context.

### TVM Runtime

TVM's runtime tensor (`tvm::runtime::NDArray`) is a lightweight handle referencing a managed `Object` container that owns storage and shape. Compiled functions (e.g., lowered graph modules) operate on these handles. This reinforces the separation between tensor handles and the underlying storage/logical tensor object.

## Scope

- **This ADR is a long-term architectural direction.**
- **Do not implement the refactor now.**
- **Do not modify current tensor semantics.**
- **Do not rewrite autograd.**

The current `input_ids_` workaround is sufficient for the existing codebase. This ticket exists to record the intended architecture before additional features (views, storage planner, compiler, CUDA, allocator improvements) increase the migration cost.

## Priority

Medium–High.

Not required for current eager execution milestones, but should be completed **before** implementing:
- Tensor views or slice operations
- Memory planner or buffer reuse pool
- Graph compilation or lowering
- CUDA backend integration
- Any feature that requires stable logical tensor identity across handle copies

## Roadmap Note

This decision supports Axon's long-term roadmap toward a PyTorch/TVM-style runtime architecture while keeping the current eager implementation stable during ongoing backend work.
