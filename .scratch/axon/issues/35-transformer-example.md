# 35 — Transformer Example (MultiHeadAttention + GPT-2 Block)

## What to build

A working Transformer example demonstrating Axon's module system with a GPT-2-style block: `MultiHeadAttention` module, `TransformerBlock` module (pre-norm), and a minimal `MiniGPT` model that can forward + backward on a small sequence.

This is an integration example, not a production model. The goal is to prove the op surface is complete and the autograd graph handles attention correctly.

## Dependencies

- **#32** — Scalar-tensor arithmetic (`/ sqrt(d_k)`)
- **#33** — Softmax as first-class op (attention weights)
- **#34** — Shape manipulation (multi-head reshape, cat/split)

## Design note

Write as an `examples/transformer.cpp` standalone file that links against `libaxon`. Modules should compose using existing `Module` / `Parameter` / `Sequential` patterns. Causal masking can be implemented as a utility function that fills upper-triangle attention scores with a large negative value before softmax.

## Acceptance criteria

- [ ] `MultiHeadAttention` module: Q/K/V projection, scaled dot-product attention, output projection
- [ ] `TransformerBlock` module: pre-LayerNorm, MHA, residual add, FFN (Linear → GELU → Linear), residual add
- [ ] `MiniGPT` model: token Embedding + positional Embedding + N TransformerBlocks + final Linear
- [ ] Forward pass produces correct output shape for `(batch=2, seq_len=16, d_model=64)`
- [ ] Backward pass completes without error and all parameter gradients are non-zero
- [ ] Causal mask prevents attention to future positions
- [ ] Example compiles and runs as a standalone CMake target

## Status

backlog (blocked by #32, #33, #34)
