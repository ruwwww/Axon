#include "axon/backend/strategy.h"

namespace axon::cpu {

GemmStrategy choose_gemm_strategy(
    const std::vector<int64_t>& shape_a,
    const std::vector<int64_t>& shape_b,
    DType dtype_a,
    QuantFormat quant_a,
    bool is_contiguous_a,
    bool is_contiguous_b,
    const CpuFeatures& features,
    bool blas_available
) {
    GemmStrategy strat;

    // 1. Quantized Matrix Multiplication Path
    if (quant_a != QuantFormat::None) {
        strat.provider = GemmProvider::AxonQuantized;
        strat.isa = (features.avx2 && features.fma3) ? ISA::AVX2 : ISA::Scalar;

        switch (quant_a) {
            case QuantFormat::Q4_0: strat.kernel_name = "matmul_q4_0"; break;
            case QuantFormat::Q4_K: strat.kernel_name = "matmul_q4_K"; break;
            case QuantFormat::Q5_K: strat.kernel_name = "matmul_q5_K"; break;
            case QuantFormat::Q8_0: strat.kernel_name = "matmul_q8_0"; break;
            case QuantFormat::Q2_K: strat.kernel_name = "matmul_q2_K"; break;
            case QuantFormat::Q3_K: strat.kernel_name = "matmul_q3_K"; break;
            case QuantFormat::Q6_K: strat.kernel_name = "matmul_q6_K"; break;
            default: strat.kernel_name = "matmul"; break;
        }
        return strat;
    }

    // 2. FP32 Contiguous Matrix Multiplication Path
    if (dtype_a == DType::Float32 && is_contiguous_a && is_contiguous_b &&
        shape_a.size() == 2 && shape_b.size() == 2) {

        int64_t M = shape_a[0];
        int64_t K = shape_a[1];
        int64_t N = shape_b[1];

        // Large FP32 matrices (> 64) delegate to External BLAS when available
        if (blas_available && M > 64 && N > 64 && K > 64) {
            strat.provider = GemmProvider::ExternalBLAS;
            strat.isa = ISA::Scalar;
            strat.kernel_name = "matmul_blas";
            return strat;
        }

        // Medium/Small FP32 matrices use Axon SIMD (AVX2) when available
        if (features.avx2 && features.fma3) {
            strat.provider = GemmProvider::AxonSIMD;
            strat.isa = ISA::AVX2;
            strat.kernel_name = "matmul";
            return strat;
        }
    }

    // 3. Fallback Path
    strat.provider = GemmProvider::AxonScalar;
    strat.isa = ISA::Scalar;
    strat.kernel_name = "matmul";
    return strat;
}

} // namespace axon::cpu
