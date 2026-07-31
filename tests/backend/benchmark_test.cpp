#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include "axon/backend/cpu_backend.h"
#include "axon/backend/registry.h"
#include "axon/runtime/runtime.h"

TEST_CASE("Benchmark elementwise AVX2 SIMD vs Scalar throughput", "[.benchmark][backend]") {
    axon::Runtime rt;
    constexpr int64_t numel = 1'000'000;
    auto a = rt.ones({numel});
    auto b = rt.ones({numel});
    auto out = rt.empty({numel});

    constexpr int iterations = 100;

    // Benchmark scalar execution
    auto fn_scalar = axon::cpu::KernelRegistry::instance().lookup("add", axon::cpu::ISA::Scalar);
    REQUIRE(fn_scalar != nullptr);

    axon::Tensor outputs[] = {out};
    axon::Tensor inputs[] = {a, b};
    axon::cpu::KernelContext ctx{.outputs = outputs, .inputs = inputs};

    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn_scalar(ctx);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

    // Benchmark AVX2 execution if supported
    if (axon::cpu::has_avx2()) {
        auto fn_avx2 = axon::cpu::KernelRegistry::instance().lookup("add", axon::cpu::ISA::AVX2);
        REQUIRE(fn_avx2 != nullptr);

        auto start_avx2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            fn_avx2(ctx);
        }
        auto end_avx2 = std::chrono::high_resolution_clock::now();
        double avx2_ms = std::chrono::duration<double, std::milli>(end_avx2 - start_avx2).count();

        double speedup = scalar_ms / (avx2_ms > 0 ? avx2_ms : 0.001);
        SUCCEED("Scalar: " + std::to_string(scalar_ms) + " ms, AVX2: " + std::to_string(avx2_ms) + " ms (Speedup: " + std::to_string(speedup) + "x)");
    }
}
