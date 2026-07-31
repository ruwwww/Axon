# 29 — OpenBLAS / BLAS GEMM Provider Integration

## What to build

Integrate optional third-party BLAS execution path (OpenBLAS / BLIS / oneDNN) under `KernelRegistry` for large FP32 matrix multiplication, complementing Axon's native SIMD kernels.

## Acceptance criteria

- [ ] Add CMake option `-DAXON_USE_BLAS=ON` to detect and link system BLAS libraries (`cblas_sgemm`)
- [ ] Register `Provider::BLAS` FP32 matmul kernel in `KernelRegistry`
- [ ] Benchmark comparison between Axon native AVX2 and OpenBLAS for $M, N, K \ge 256$
- [ ] All unit test cases pass cleanly

## Status

backlog
