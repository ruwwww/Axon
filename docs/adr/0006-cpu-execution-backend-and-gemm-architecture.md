# ADR-0006: Hybrid CPU Execution Backend & GEMM Architectural Direction

## Context

Axon requires high-performance matrix multiplication (GEMM) and elementwise tensor operations across CPU and future hardware accelerators.

Building a full production-grade BLIS/oneDNN replacement from scratch (with cache-panel packing, L2/L3 cache blocking, and architecture-tuned assembly microkernels) would impose an unmaintainable maintenance burden. Conversely, relying solely on third-party BLAS libraries would obscure hardware-level learning and prevent executing specialized deep learning workloads (such as $Q4_0$, $Q4_K$, and $Q5_K$ block-quantized weight dot products or fused `MatMul + Bias + Activation` kernels).

## Decision

We adopt a **Hybrid CPU Execution Backend Architecture** matching the patterns of PyTorch ATen, oneDNN, and TVM:

1. **Hybrid Execution Seam**:
   - **External BLAS Path (OpenBLAS / BLIS / oneDNN)**: High-performance FP32 GEMM for large matrices ($M, N, K > 64$) delegates to third-party BLAS providers.
   - **Axon Native SIMD Kernels**: Native C++ AVX2/FMA SIMD kernels serve as: (1) Reference and educational implementations; (2) Fallback for zero-dependency environments; and (3) Execution engine for specialized quantized block formats ($Q4_0, Q4_K, Q5_K$) and fused operators.

2. **GEMM Classification**:
   - Axon's initial native AVX2 GEMM kernel is classified as **Vectorized Naive GEMM** (SIMD $1 \times 8$ register accumulation without cache-panel packing). Future optimization stages (Cache Blocking ➔ Packed GEMM ➔ Microkernel) will be introduced strictly when driven by benchmark metrics.

3. **Dispatcher & Kernel Seam Evolution**:
   - All execution kernels conform to a generic `KernelContext` interface (`outputs`, `inputs`, `attributes`), providing an execution seam for both eager runtime calls and future graph compiler JIT lowerings.
   - `KernelRegistry` dispatch keys will evolve from string `"op_name:isa"` pairs to typed composite keys `(OpId, Device, DType, Layout, ISA, Provider)`.
   - Kernel selection for GEMM is governed by a dynamic strategy policy (`choose_gemm_strategy(shape, dtype, layout, features)`) rather than hardcoded size thresholds.

## Consequences

- **Performance**: Peak FLOPS achieved on large standard FP32 GEMMs via external BLAS while retaining zero latency on small matrices via native SIMD loops.
- **Flexibility**: Specialized LLM quantized formats ($Q4_0, Q4_K, Q5_K$) run efficiently through native SIMD unpacking kernels.
- **Portability**: Codebase remains 100% buildable and executable without external dependencies.
