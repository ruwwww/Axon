# 37 — Rename TensorType to TensorMetadata

## What to build

Rename `TensorType` to `TensorMetadata` across the entire codebase. This is a naming and architectural clarification only — no behavioral changes.

## Motivation

The current name `TensorType` overloads the meaning of "type." In compiler frameworks (MLIR, XLA, TVM IR), a **TensorType** is an IR/type-system concept. Axon's current object instead represents **runtime metadata** attached to a tensor:

* shape
* strides
* dtype
* device
* layout
* storage offset
* quantization metadata (if applicable)

These are runtime properties, not compiler types.

To avoid future confusion between runtime and compiler concepts, rename the runtime descriptor to **TensorMetadata**.

## Architectural Intent

Long-term runtime hierarchy:

```
Tensor
    ↓
TensorImpl
    ├── TensorMetadata
    ├── Storage
    ├── Autograd state
    └── Version / runtime flags
```

Future compiler hierarchy (separate subsystem):

```
Graph IR
    ↓
Compiler TensorType
    ↓
Lowering
```

These are intentionally different concepts.

## Scope

* Rename `TensorType` → `TensorMetadata` everywhere: headers, sources, tests, docs, ADRs, comments
* Update constructors, APIs, documentation, and comments
* Preserve behavior — this is a naming/architectural clarification only
* Do **not** change the storage model or ownership semantics
* Do **not** introduce compiler-specific abstractions at this stage

## Files to update (indicative)

* `include/axon/tensor/tensor_type.h` → `tensor_metadata.h`
* All `#include "axon/tensor/tensor_type.h"` references
* All `TensorType::` usage sites
* All `TensorType` variable/parameter/return types
* Documentation: `CONTEXT.md`, `docs/adr/*.md`
* Tests: `tests/tensor/tensor_iterator_test.cpp`, `tests/backend/cpu_backend_test.cpp`, and any other test referencing `TensorType`

## Acceptance criteria

- [ ] Header renamed to `include/axon/tensor/tensor_metadata.h`
- [ ] All source files updated to use `TensorMetadata`
- [ ] All tests updated and pass
- [ ] Documentation and ADRs updated
- [ ] No behavioral changes — all 213 tests pass with identical results

## Status

backlog
