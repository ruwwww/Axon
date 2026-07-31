# 33 — Softmax as First-Class Runtime Operation

## What to build

Promote `cpu::softmax` from an internal backend function (currently only used inside CrossEntropyLoss) to a full autograd-aware operation.

Attention scores require `softmax(Q @ K^T / sqrt(d_k))` with gradients flowing back through the softmax. This is the core blocker for any attention mechanism.

## Design note

Implement `SoftmaxNode` in the autograd layer with correct Jacobian: `grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))`. The backend `cpu::softmax` already exists and is correct. The Runtime wiring is the current API surface but will eventually migrate to `Tensor.softmax(dim)`.

## Acceptance criteria

- [ ] `SoftmaxNode` autograd node with correct backward (Jacobian-vector product)
- [ ] `runtime.softmax(x)` wired through Operation → backend → autograd
- [ ] Forward matches existing `cpu::softmax` output
- [ ] Backward verified against finite differences on small tensors
- [ ] Unit tests (forward correctness, backward gradient check)
- [ ] All existing tests still pass

## Status

backlog
