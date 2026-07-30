# 22 — AVX2 & AVX-VNNI SIMD Vectorized CPU Kernels

## What to build

Accelerate Axon's CPU backend by introducing AVX2 / FMA3 and AVX-VNNI (Vector Neural Network Instructions) SIMD vector intrinsics for element-wise operations, FP32 matrix multiplication, and on-the-fly quantized block dot-products (`Q4_0`, `Q4_K`, `Q5_K`).

Target host capabilities available on Intel Core 13th Gen (and modern x86_64 CPUs) to replace scalar loops with 256-bit SIMD vector operations (`_mm256_fmadd_ps`, `_mm256_dpbusd_epi32`).

## Blocked by

None — can start immediately.

## Acceptance criteria

- [ ] CPU backend detects runtime AVX2 / AVX-VNNI CPU feature support via CPUID
- [ ] FP32 matmul utilizes AVX2 FMA vector SIMD loops when available with scalar fallback
- [ ] Quantized matmul kernels (`Q4_0`, `Q4_K`, `Q5_K`) use SIMD vector unpacking & dot-product accumulation
- [ ] All existing test cases pass unchanged and produce bit-exact / within-tolerance results
- [ ] Benchmark shows $\ge 4\times$ throughput speedup on AVX2-capable host CPUs

## Status

ready-for-agent
