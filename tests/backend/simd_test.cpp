#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "axon/backend/simd/vec.h"
#include "axon/backend/simd/scalar.h"
#include "axon/backend/simd/avx2.h"

TEST_CASE("Scalar Vec<float> basic arithmetic operations", "[simd][scalar]") {
    float a[] = {3.0f};
    float b[] = {4.0f};
    float out[1] = {0.0f};

    auto va = axon::simd::Vec<float, axon::cpu::ISA::Scalar>::load(a);
    auto vb = axon::simd::Vec<float, axon::cpu::ISA::Scalar>::load(b);
    auto vc = va * vb + axon::simd::Vec<float, axon::cpu::ISA::Scalar>::set1(2.0f);
    vc.store(out);

    REQUIRE_THAT(out[0], Catch::Matchers::WithinRel(14.0f, 1e-5f));
}

#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86)))
TEST_CASE("AVX2 Vec<float> 8-wide SIMD arithmetic and fmadd", "[simd][avx2]") {
    alignas(32) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) float b[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    alignas(32) float c[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    alignas(32) float out[8] = {0};

    auto va = axon::simd::Vec<float, axon::cpu::ISA::AVX2>::load(a);
    auto vb = axon::simd::Vec<float, axon::cpu::ISA::AVX2>::load(b);
    auto vc = axon::simd::Vec<float, axon::cpu::ISA::AVX2>::load(c);

    auto res = axon::simd::Vec<float, axon::cpu::ISA::AVX2>::fmadd(va, vb, vc);
    res.store(out);

    for (int i = 0; i < 8; ++i) {
        REQUIRE_THAT(out[i], Catch::Matchers::WithinRel(a[i] * 2.0f + 1.0f, 1e-5f));
    }
}
#endif
