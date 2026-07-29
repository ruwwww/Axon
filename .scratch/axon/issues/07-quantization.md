# 07 — Quantization (M5)

## Context

Axon is a minimal deep learning framework. This ticket adds GGML-inspired low-precision tensor storage and computation. Read `SPEC.md`, `CONTEXT.md`, and `docs/adr/*` for the full architecture. Ticket 01 must be complete (needs Storage foundation).

## What to build

Quantized tensor storage and computation kernels, inspired by GGML block layouts:

- **QuantizationDescriptor**: Defines the encoding scheme per format:
  - block_size (e.g., 32 elements per block)
  - packing (how values are packed into bytes)
  - scales (per-block scale factors)
  - decode rule (function to reconstruct float values)
- **Quantized Storage**: Storage with a QuantizationDescriptor set. Byte layout is the packed representation, not the dequantized values. The raw byte count reflects the compressed size.
- **Quantization formats** (from the spec):
  - Q8_0: 8-bit symmetric per block
  - Q6_K: 6-bit, K-quant
  - Q5_K, Q4_0, Q4_K, Q3_K, Q2_K
- **Quantized CPU backend kernels**: `cpu::matmul_q4(out, a, b)`, `cpu::relu_q8(out, x)` etc. — kernels that operate directly on packed storage without materializing the full float32 tensor.
- **Quantize/Dequantize functions**: `cpu::quantize(src, dst, quant_type)` converts a float32 Storage to quantized. `cpu::dequantize(src, dst)` converts back for debugging.
- **Example**: `examples/quantized_inference.cpp` — loads a model, quantizes weights, runs inference, compares output to float32 version within tolerance.

## Key interfaces

```cpp
// QuantizationDescriptor
struct QuantizationDescriptor {
    QuantFormat format;  // Q8_0, Q6_K, Q5_K, Q4_0, ...
    size_t block_size;
    // format-specific metadata (e.g., scale_bits, have_min, etc.)
};

enum class QuantFormat {
    None, Q8_0, Q6_K, Q5_K, Q4_0, Q4_K, Q3_K, Q2_K
};

// CPU backend additions
namespace cpu {
    size_t quantized_size(size_t num_elements, QuantFormat format);
    Expected<void> quantize(const Tensor& src, Tensor& dst, QuantFormat format);
    Expected<void> dequantize(const Tensor& src, Tensor& dst);
    Expected<void> matmul_q4(Tensor& out, const Tensor& a, const Tensor& b);
}
```

## Acceptance criteria

- [ ] QuantizationDescriptor correctly computes compressed size for each format.
- [ ] Quantize → Dequantize roundtrip preserves values within the expected precision loss.
- [ ] Quantized matmul kernel (e.g., Q4_0) produces results within tolerance of float32 matmul.
- [ ] A quantized model loads and runs inference without materializing full float32 weights.
- [ ] M5 example compiles and produces reasonable results.

## Blocked by

01 — Foundation: Storage + Tensor + Allocator + CPU Backend (needs Storage, but can be parallel with tickets 02–06)

## Status

ready-for-agent
