# Axon Execution Runtime Roadmap

This document tracks the long-term execution roadmap for Axon. Individual work items are filed as tickets in `.scratch/axon/issues/`.

---

## 🗺️ Execution Phases

```
Phase 1: Eager Runtime Foundation (COMPLETED)
  ├── Tensor, Storage, Autograd DAG (Node), KernelRegistry
  └── Vectorized Naive GEMM, SIMD traits, CPUID (Tickets #21, #22, #24, #25)

Phase 2: Backend Provider Architecture
  └── Streamlined KernelKey (OpId, Device, DType, Provider) & CPU/CUDA separation (Ticket #28)

Phase 3: Memory System & Buffer Reuse
  └── Allocator, views, strides, contiguous checks, buffer reuse pool

Phase 4: Production CPU Backend & BLAS
  ├── OpenBLAS / BLIS / oneDNN integration (Ticket #29)
  └── Strategy policy choose_gemm_strategy(...) (Ticket #30)

Phase 5: Permanent Benchmark Infrastructure
  └── Automated benchmark suite for Small, Medium, Large matrices (Ticket #31)

Phase 6: Specialized ML Kernels
  └── Quantized inference (Q4_0, Q4_K, Q5_K), fused kernels, attention (Ticket #26)

Phase 7: Graph Runtime & Compiler/JIT
  └── Graph IR, operation fusion, memory planning, and lowered code generation
```

---

## 📋 Open Backlog Tickets

- [x] **#21** — Polymorphic Node DAG Autograd Engine
- [x] **#22** — AVX2 SIMD CPU Backend & Architecture Assessment
- [x] **#26** — AVX-VNNI & Advanced Quantized SIMD Kernels ($Q4_K, Q5_K$)
- [ ] **#27** — KernelRegistry Device Dimension & Storage Accessor Seam
- [ ] **#28** — Streamlined KernelKey (`OpId`, `Device`, `DType`, `Provider`) & Provider Abstraction
- [ ] **#29** — OpenBLAS / BLAS GEMM Provider Integration
- [ ] **#30** — Dynamic GEMM Strategy Selection Policy
- [ ] **#31** — Permanent Automated Benchmark Suite
