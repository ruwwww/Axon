#include <catch2/catch_test_macros.hpp>
#include "axon/tensor/tensor.h"
#include "axon/runtime/runtime.h"

TEST_CASE("const_cast UB analysis: modifying mutable memory via const ref is safe", "[ub][batchnorm]") {
    axon::Runtime rt;

    // Create a mutable Tensor (heap-allocated storage, never declared const)
    axon::Tensor mutable_tensor = axon::Tensor::ones(rt, {4});

    // Verify initial values are 1.0
    for (int i = 0; i < 4; ++i) {
        REQUIRE(mutable_tensor.data<float>()[i] == 1.0f);
    }

    // Take a const reference to the same tensor (this is what BatchNormOp::forward does)
    const axon::Tensor& const_ref = mutable_tensor;

    // Simulating what cpu::batchnorm does:
    // 1. Gets const float* via data<const float>() on a const ref
    const float* const_ptr = const_ref.data<const float>();

    // 2. const_casts away const (exactly like cpu_backend.cpp line 367-368)
    float* mut_ptr = const_cast<float*>(const_ptr);

    // 3. Writes through the cast pointer (exactly like cpu_backend.cpp line 370-371)
    for (int i = 0; i < 4; ++i) {
        mut_ptr[i] = 42.0f;
    }

    // 4. Verify changes are visible through the original mutable tensor
    for (int i = 0; i < 4; ++i) {
        REQUIRE(mutable_tensor.data<float>()[i] == 42.0f);
    }

    // 5. Verify changes are visible through the const ref too
    for (int i = 0; i < 4; ++i) {
        REQUIRE(const_ref.data<const float>()[i] == 42.0f);
    }
}

TEST_CASE("Tensor::data() bypasses const entirely - const_cast is redundant", "[ub][batchnorm]") {
    axon::Runtime rt;

    axon::Tensor t = axon::Tensor::zeros(rt, {3});
    const axon::Tensor& ct = t;

    // Key observation: Tensor::data<T>() is a const member function that returns T*
    // So data<float>() on a const Tensor& STILL returns float* (non-const)
    // This means the const_cast in cpu_backend.cpp is completely unnecessary
    float* direct = ct.data<float>();  // Works on const ref! No const_cast needed.

    direct[0] = 99.0f;
    REQUIRE(t.data<float>()[0] == 99.0f);
}

TEST_CASE("Genuinely const-declared Tensor is still safe due to heap storage", "[ub][batchnorm]") {
    axon::Runtime rt;

    // Even if the Tensor itself is declared const, the underlying storage
    // is heap-allocated (via _aligned_malloc), so the memory is mutable.
    // The const only applies to the Tensor object's members, not the pointed-to data.
    const axon::Tensor ct = axon::Tensor::ones(rt, {3});

    // data<float>() works because it's a const member function
    // and the storage pointer itself isn't const
    float* p = ct.data<float>();
    p[0] = 77.0f;

    REQUIRE(ct.data<float>()[0] == 77.0f);
    // This is NOT UB because the float values are in heap memory,
    // never declared const. Only the Tensor object (id_, storage_, type_ pointers) is const.
}
