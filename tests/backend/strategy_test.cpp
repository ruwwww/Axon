#include <catch2/catch_test_macros.hpp>
#include "axon/backend/strategy.h"

TEST_CASE("choose_gemm_strategy routes small, large, and quantized workloads correctly", "[backend][strategy]") {
    axon::cpu::CpuFeatures mock_avx2{.avx = true, .avx2 = true, .fma3 = true};
    axon::cpu::CpuFeatures mock_scalar{};

    int64_t s16[] = {16, 16};
    int64_t s128[] = {128, 128};
    int64_t s32_128[] = {32, 128};
    int64_t s128_32[] = {128, 32};
    int64_t s32_256[] = {32, 256};
    int64_t s256_32[] = {256, 32};

    SECTION("Small FP32 matrices (M,N,K <= 64) route to Axon SIMD even when BLAS is available") {
        auto strat = axon::cpu::choose_gemm_strategy(
            s16, s16,
            axon::DType::Float32, axon::QuantFormat::None,
            true, true, mock_avx2, true
        );
        REQUIRE(strat.provider == axon::cpu::GemmProvider::AxonSIMD);
        REQUIRE(strat.isa == axon::cpu::ISA::AVX2);
        REQUIRE(strat.kernel_name == "matmul");
    }

    SECTION("Large FP32 matrices (M,N,K > 64) route to External BLAS when BLAS is available") {
        auto strat = axon::cpu::choose_gemm_strategy(
            s128, s128,
            axon::DType::Float32, axon::QuantFormat::None,
            true, true, mock_avx2, true
        );
        REQUIRE(strat.provider == axon::cpu::GemmProvider::ExternalBLAS);
        REQUIRE(strat.kernel_name == "matmul_blas");
    }

    SECTION("Large FP32 matrices route to Axon SIMD when BLAS is NOT available") {
        auto strat = axon::cpu::choose_gemm_strategy(
            s128, s128,
            axon::DType::Float32, axon::QuantFormat::None,
            true, true, mock_avx2, false
        );
        REQUIRE(strat.provider == axon::cpu::GemmProvider::AxonSIMD);
        REQUIRE(strat.isa == axon::cpu::ISA::AVX2);
        REQUIRE(strat.kernel_name == "matmul");
    }

    SECTION("Quantized tensors route to Axon Quantized provider") {
        auto strat_q4 = axon::cpu::choose_gemm_strategy(
            s32_128, s128_32,
            axon::DType::Float32, axon::QuantFormat::Q4_0,
            true, true, mock_avx2, true
        );
        REQUIRE(strat_q4.provider == axon::cpu::GemmProvider::AxonQuantized);
        REQUIRE(strat_q4.kernel_name == "matmul_q4_0");

        auto strat_q4K = axon::cpu::choose_gemm_strategy(
            s32_256, s256_32,
            axon::DType::Float32, axon::QuantFormat::Q4_K,
            true, true, mock_avx2, true
        );
        REQUIRE(strat_q4K.provider == axon::cpu::GemmProvider::AxonQuantized);
        REQUIRE(strat_q4K.kernel_name == "matmul_q4_K");
    }

    SECTION("Scalar fallback for non-AVX2 host hardware") {
        auto strat = axon::cpu::choose_gemm_strategy(
            s16, s16,
            axon::DType::Float32, axon::QuantFormat::None,
            true, true, mock_scalar, false
        );
        REQUIRE(strat.provider == axon::cpu::GemmProvider::AxonScalar);
        REQUIRE(strat.isa == axon::cpu::ISA::Scalar);
        REQUIRE(strat.kernel_name == "matmul");
    }
}
