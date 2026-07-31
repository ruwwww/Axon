# ADR-001: Tensor Identity in Autograd Nodes

## Status

Proposed

## Context

Autograd `Node` subclasses store their input tensors by value in `std::vector<Tensor> inputs_`. The `Tensor` class uses value semantics (copyable, movable), but each instance carries a unique `TensorId` assigned at construction.

When an operation's `forward()` creates a `Node` and passes input tensors by value, those tensors are moved into the node. After the move, the original tensor objects in the caller's scope are in a valid but unspecified state. More critically, if a copy is made anywhere in the chain, the copy gets a new `TensorId`.

The `GradientMap` (used during backward) is keyed by `TensorId`. If a `Node::apply()` looks up gradients using the ID of a copied/moved tensor instead of the original input tensor's ID, the lookup fails silently — the gradient is never found, and backward returns early with no gradients stored.

## Discovery

This bug manifested during ticket #32 (scalar-tensor arithmetic ops). All backward tests for `SubOp`, `MulScalarOp`, and `DivScalarOp` failed with empty gradient maps. The root cause was that `Node::inputs_` held copied tensors with different IDs than the originals stored in `GradientMap`.

Debug prints were added to trace gradient lookups, but were not visible in the Catch2 test runner output, making diagnosis slow.

## Decision

Short-term fix: Each `Node` subclass stores a parallel `std::vector<TensorId> input_ids_` initialized from the input tensors' IDs *before* they are moved into `inputs_`. A new pure virtual `input_ids()` method on `Node` returns these stored IDs. `Autograd::backward()` uses `input_ids()` instead of `inputs_[i].id()` for gradient map lookups.

This is a workaround, not a fix. It duplicates identity information that already exists in the tensor objects, and relies on every new `Node` subclass remembering to:
1. Capture IDs before moving tensors
2. Store them in `input_ids_`
3. Implement `input_ids()`

Missing any step causes silent gradient loss.

## Consequences

- **Positive**: Backward propagation works correctly. All 213 tests pass.
- **Negative**: The `input_ids_` pattern is fragile and must be replicated for every new `Node` subclass.
- **Negative**: Silent failures are possible if the pattern is not followed exactly.

## Future Improvements

1. **Make `Tensor` non-copyable**: Force all graph communication to use references or `Tensor*` pointers. This eliminates the ID-change-on-copy problem at the type level.
2. **Use `TensorId` directly in `Node::inputs_`**: Store input IDs in the node, and look up tensors from the `GradientMap` or a separate tensor registry during backward. This decouples node storage from tensor identity.
3. **Add validation in `Autograd::backward()`**: After the backward pass, verify that all tensors with `requires_grad=true` have non-empty `.grad`. Log a warning (or error) if any are missing. This would have caught the bug immediately instead of requiring debug prints.
4. **Replace debug prints with structured logging**: Add a `verbose` flag to `Autograd` that controls diagnostic output, rather than ad-hoc `fprintf` calls.

## Lessons Learned

- When a data structure carries identity (like `TensorId`), value semantics (copying, moving) can silently break identity-dependent systems.
- Silent failures are far more expensive than loud ones. The gradient map returning empty should have been an error condition, not a silent early-return.
- Test construction matters: tests that create a separate loss tensor with a different ID than the graph output will hit this bug, even when the backward implementation is otherwise correct.
