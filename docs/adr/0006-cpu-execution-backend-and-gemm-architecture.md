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

3. **Primacy of the Memory Model**:
   - Memory model development (contiguous checks, strides, non-owning tensor views, allocator, and buffer reuse pool) is prioritized before external providers and graph compilers, because memory layout governs kernel selection, performance, and fusion.

4. **Streamlined `KernelKey` Design**:
   - Dispatch keys match PyTorch ATen / oneDNN levels of granularity: `KernelKey: (OpId, Device, DType, Provider)`. The CPU provider internally selects `ISA::AVX2` or `ISA::Scalar`.

## Revised Long-Term Roadmap

```
Phase 1: Eager Runtime Foundation (COMPLETED)
  └── Tensor, Autograd DAG, Vectorized Naive GEMM, SIMD traits, CPUID

Phase 2: Backend Provider Architecture (Ticket #28)
  └── Streamlined KernelKey (OpId, Device, DType, Provider) & provider abstraction

Phase 3: Memory System & Buffer Reuse
  └── Allocator, views, strides, contiguous checks, buffer reuse pool

Phase 4: Production CPU Backend & BLAS (Tickets #29 & #30)
  └── OpenBLAS / BLIS / oneDNN integration & strategy policy choose_gemm_strategy(...)

Phase 5: Permanent Benchmark Infrastructure (Ticket #31)
  └── Comprehensive benchmark suite (Small, Medium, Large matrices; Scalar vs AVX2 vs BLAS)

Phase 6: Specialized ML Kernels (Ticket #26)
  └── Quantized inference (Q4_0, Q4_K, Q5_K), fused kernels, and attention

Phase 7: Graph Runtime & Compiler/JIT
  └── Graph IR, operation fusion, memory planning, and lowered code generation
```

## Consequences

- **Architectural Clarity**: Clear separation between problem description, provider selection, and kernel execution.
- **Production Performance**: Peak FLOPS achieved on large standard FP32 GEMMs via external BLAS while retaining low latency on small matrices via native SIMD loops.
- **Maintainability**: `KernelKey` remains clean without over-engineering every lookup dimension.
