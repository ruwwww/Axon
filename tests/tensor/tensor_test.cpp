#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/tensor/tensor.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Tensor::empty creates tensor with correct shape", "[tensor]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {2, 3});
    REQUIRE(t.defined());
    REQUIRE(t.type().shape() == std::vector<int64_t>({2, 3}));
    REQUIRE(t.type().dtype() == DType::Float32);
}

TEST_CASE("Tensor::zeros creates tensor filled with zeros", "[tensor]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {2, 3});
    REQUIRE(t.defined());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(0.0f));
    }
}

TEST_CASE("Tensor::ones creates tensor filled with ones", "[tensor]") {
    Runtime rt;
    auto t = Tensor::ones(rt, {2, 3});
    REQUIRE(t.defined());
    auto* data = t.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(1.0f));
    }
}

TEST_CASE("Tensor copy shares storage", "[tensor][refcount]") {
    Runtime rt;
    auto t1 = Tensor::zeros(rt, {2, 3});
    auto t2 = t1;

    REQUIRE(t1.storage() == t2.storage());
    REQUIRE(t1.storage().use_count() >= 2);
}

TEST_CASE("Tensor::randn creates tensor with correct shape and dtype", "[tensor]") {
    Runtime rt;
    auto t = Tensor::randn(rt, {4, 5});
    REQUIRE(t.defined());
    REQUIRE(t.type().shape() == std::vector<int64_t>({4, 5}));
    REQUIRE(t.type().dtype() == DType::Float32);
}

TEST_CASE("Tensor::empty allocates correct size for float32", "[tensor]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {10, 20});
    REQUIRE(t.storage()->size_bytes == 10 * 20 * 4);
}
