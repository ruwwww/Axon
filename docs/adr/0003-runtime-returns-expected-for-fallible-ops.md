# Runtime convenience methods return Expected<Tensor> for fallible operations

Runtime::matmul and Runtime::relu return `Expected<Tensor>` rather than the plain `Tensor` sketched in the original ticket interface. This aligns with the project-wide error-handling standard (ADR-0001) that all cross-subsystem public APIs return `Expected<T>` or `Status`.

The earlier creation methods (empty, zeros, ones, randn) return plain `Tensor` because they are infallible. The operation methods are fallible (shape mismatch, dtype mismatch) so they use `Expected<T>`.

This is a deviation from the interface shown in ticket 02, which showed plain `Tensor matmul(...)` and `Tensor relu(...)` as shorthand. The actual implementation uses the safer return type. Future ticket sketches should prefer `Expected<Tensor>` for fallible operations.

## Backward skips nodes without a gradient in the map

During backward traversal, if a Node's output has no entry in the GradientMap (because it is not reachable from the loss), the node is silently skipped rather than returning an error. This enables correct partial-graph backward: when the graph branches and backward is called on one branch, the other branch is harmless to skip.

The original implementation returned an error, which would incorrectly reject valid partial traversals.
