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

TEST_CASE("Allocator-created tensor has storage_offset == 0", "[tensor][offset]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {2, 3});
    REQUIRE(t.storage_offset() == 0);

    auto t2 = Tensor::empty(rt, {4, 5, 6});
    REQUIRE(t2.storage_offset() == 0);
}

TEST_CASE("Tensor data<T>() with non-zero offset points past start of storage", "[tensor][offset]") {
    Runtime rt;
    auto storage = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto* raw = static_cast<float*>(storage->data);
    for (int i = 0; i < 6; ++i) raw[i] = static_cast<float>(i + 1);

    // Create tensor with offset 3 (offset by 3 elements)
    TensorType type({3}, {1}, DType::Float32);
    Tensor t(type, storage, false, 3);

    REQUIRE(t.storage_offset() == 3);
    REQUIRE(t.data<float>()[0] == Catch::Approx(4.0f));
    REQUIRE(t.data<float>()[1] == Catch::Approx(5.0f));
    REQUIRE(t.data<float>()[2] == Catch::Approx(6.0f));

    // Original tensor at offset 0
    Tensor t0(type, storage, false, 0);
    REQUIRE(t0.data<float>()[0] == Catch::Approx(1.0f));
    REQUIRE(t0.data<float>()[2] == Catch::Approx(3.0f));
}
