#include <catch2/catch_test_macros.hpp>
#include "axon/backend/cpuid.h"

TEST_CASE("CPU feature detection queries host hardware without crashing", "[backend][cpuid]") {
    auto features = axon::cpu::detect_cpu_features();
    auto best_isa = axon::cpu::get_best_isa();

    // Verification: cpuid execution succeeds and returns valid enum
    REQUIRE((best_isa == axon::cpu::ISA::Scalar || best_isa == axon::cpu::ISA::AVX2));
    if (best_isa == axon::cpu::ISA::AVX2) {
        REQUIRE(features.avx2);
        REQUIRE(features.fma3);
        REQUIRE(axon::cpu::has_avx_vnni() == features.avx_vnni);
    }
}
