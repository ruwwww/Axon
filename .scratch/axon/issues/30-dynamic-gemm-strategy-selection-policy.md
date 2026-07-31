# 30 — Dynamic GEMM Strategy Selection Policy

## What to build

Implement a heuristic strategy selection function `choose_gemm_strategy(shape, dtype, layout, cpu_features)` in the CPU backend to dynamically choose between BLAS, Axon AVX2 SIMD, and Axon Quantized kernels based on dimensions, memory contiguity, and host hardware capabilities.

## Acceptance criteria

- [x] Implement `GemmStrategy choose_gemm_strategy(...)` heuristic policy
- [x] Route small $M, N, K \le 64$ to Axon AVX2 SIMD kernel to avoid library call overhead
- [x] Route large FP32 $M, N, K > 64$ to BLAS provider when available
- [x] Route $Q4_0 / Q4_K / Q5_K$ quantized tensors to Axon native quantized kernels
- [x] All unit test cases pass cleanly

## Status

completed
