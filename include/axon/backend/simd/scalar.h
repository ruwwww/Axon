#pragma once

#include "axon/backend/simd/vec.h"

namespace axon::simd {

template<>
struct Vec<float, cpu::ISA::Scalar> {
    static constexpr int size = 1;
    float val = 0.0f;

    static Vec load(const float* ptr) { return Vec{*ptr}; }
    void store(float* ptr) const { *ptr = val; }

    static Vec set1(float x) { return Vec{x}; }

    static Vec fmadd(Vec a, Vec b, Vec c) { return Vec{a.val * b.val + c.val}; }

    Vec operator+(Vec o) const { return Vec{val + o.val}; }
    Vec operator*(Vec o) const { return Vec{val * o.val}; }
    Vec operator-(Vec o) const { return Vec{val - o.val}; }
    Vec operator/(Vec o) const { return Vec{val / o.val}; }

    Vec& operator+=(Vec o) { val += o.val; return *this; }
    Vec& operator*=(Vec o) { val *= o.val; return *this; }
};

} // namespace axon::simd
