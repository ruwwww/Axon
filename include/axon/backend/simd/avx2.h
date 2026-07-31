#pragma once

#include "axon/backend/simd/vec.h"

#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86)))
#include <immintrin.h>

namespace axon::simd {

template<>
struct Vec<float, cpu::ISA::AVX2> {
    static constexpr int size = 8;
    __m256 reg;

    static Vec load(const float* ptr) { return Vec{_mm256_loadu_ps(ptr)}; }
    void store(float* ptr) const { _mm256_storeu_ps(ptr, reg); }

    static Vec set1(float x) { return Vec{_mm256_set1_ps(x)}; }

    static Vec fmadd(Vec a, Vec b, Vec c) {
        return Vec{_mm256_fmadd_ps(a.reg, b.reg, c.reg)};
    }

    Vec operator+(Vec o) const { return Vec{_mm256_add_ps(reg, o.reg)}; }
    Vec operator*(Vec o) const { return Vec{_mm256_mul_ps(reg, o.reg)}; }
    Vec operator-(Vec o) const { return Vec{_mm256_sub_ps(reg, o.reg)}; }
    Vec operator/(Vec o) const { return Vec{_mm256_div_ps(reg, o.reg)}; }

    Vec max(Vec o) const { return Vec{_mm256_max_ps(reg, o.reg)}; }

    Vec& operator+=(Vec o) { reg = _mm256_add_ps(reg, o.reg); return *this; }
    Vec& operator*=(Vec o) { reg = _mm256_mul_ps(reg, o.reg); return *this; }
};

} // namespace axon::simd

#endif
