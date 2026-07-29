# 18 — Additional quantization formats (Q6_K, Q5_K, Q4_K, Q3_K, Q2_K)

## What to build

Implement the remaining GGML-inspired block quantization formats alongside the existing Q8_0 and Q4_0:

- **Q6_K**: 6-bit, K-quant — 256-element blocks with 2 fp16 scales + 6-bit values packed into bytes, with 2-bit quants for the 4 outlier sub-blocks
- **Q5_K**: 5-bit, K-quant — 256-element blocks with super-block scaling
- **Q4_K**: 4-bit, K-quant — 256-element blocks (more complex packing than Q4_0's simple 32-element blocks)
- **Q3_K**: 3-bit, K-quant — 256-element blocks with 2 fp16 scales
- **Q2_K**: 2-bit, K-quant — 256-element blocks

For each format:
- Block struct with bitfield/packed layout matching GGML
- `quantize()` and `dequantize()` functions
- `quantized_size()` calculation
- At minimum, a quantized matmul kernel or on-the-fly dequant matmul

## Blocked by

None — can start immediately (extends existing quantization infrastructure).

## Acceptance criteria

- [ ] Each format quantize/dequantize roundtrips within expected precision loss
- [ ] quantized_size() returns correct byte count for each format
- [ ] Quantized matmul with each format produces results within tolerance of float32 matmul
- [ ] Tests for at least Q4_K and Q6_K (the most commonly used)

## Status

ready-for-agent
