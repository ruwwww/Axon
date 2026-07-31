# 31 — Permanent Automated Benchmark Suite

## What to build

Implement a dedicated, organized benchmark suite (`tests/backend/benchmark_suite_test.cpp` or `benchmarks/`) measuring performance across problem sizes (small 32x32, medium 256x256, large 2048x2048) and tracking Scalar vs AVX2 SIMD vs BLAS execution throughput.

## Acceptance criteria

- [ ] Dedicated benchmark target in CMake (`axon_benchmarks`)
- [ ] Matrix multiplication benchmarks across Small ($32 \times 32$), Medium ($256 \times 256$), and Large ($2048 \times 2048$)
- [ ] Elementwise benchmarks (`Add`, `Mul`, `ReLU`)
- [ ] Quantized dot-product benchmarks ($Q4_0$)
- [ ] Automated execution report outputting GFLOPS / GB/s metrics

## Status

backlog
