#pragma once

#include <cstdint>
#include <span>
#include <string>
#include "axon/backend/cpuid.h"
#include "axon/core/types.h"

namespace axon::cpu {

constexpr int64_t kBlasMinDimensionThreshold = 64;

enum class GemmProvider : uint8_t {
    AxonScalar = 0,
    AxonSIMD = 1,
    AxonQuantized = 2,
    ExternalBLAS = 3
};

struct GemmStrategy {
    GemmProvider provider = GemmProvider::AxonScalar;
    ISA isa = ISA::Scalar;
    std::string kernel_name = "matmul";
};

GemmStrategy choose_gemm_strategy(
    std::span<const int64_t> shape_a,
    std::span<const int64_t> shape_b,
    DType dtype_a = DType::Float32,
    QuantFormat quant_a = QuantFormat::None,
    bool is_contiguous_a = true,
    bool is_contiguous_b = true,
    const CpuFeatures& features = detect_cpu_features(),
    bool blas_available = false
);

} // namespace axon::cpu
