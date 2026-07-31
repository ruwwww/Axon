# 26 — AVX-VNNI & Advanced Quantized SIMD Kernels (Q4_K, Q5_K)

## What to build

Expand Axon's SIMD backend with AVX-VNNI (Vector Neural Network Instructions) and AVX2 vector unpacking for 256-element block quantization formats (`Q4_K`, `Q5_K`).

## Acceptance criteria

- [x] Implement AVX2 vectorized block unpacking for `Q4_K` and `Q5_K` matmul kernels
- [x] Implement AVX-VNNI uint8/int8 dot product intrinsics query (`has_avx_vnni()`) when hardware CPUID leaf 7 subleaf 1 EAX bit 4 is set
- [x] Benchmark shows throughput speedup on Q4_K / Q5_K workloads
- [x] All unit tests pass cleanly

## Status

completed
