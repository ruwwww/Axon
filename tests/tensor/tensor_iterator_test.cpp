#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include "axon/tensor/tensor_iterator.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("TensorIterator is_contiguous returns true for contiguous tensor", "[tensor_iterator]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {2, 3});
    TensorIterator<float> it(t);
    REQUIRE(it.is_contiguous());
}

TEST_CASE("TensorIterator is_contiguous returns false for transposed tensor", "[tensor_iterator]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    auto t = *rt.transpose(x, 0, 1);
    TensorIterator<float> it(t);
    REQUIRE_FALSE(it.is_contiguous());
}

TEST_CASE("TensorIterator accessors return correct values", "[tensor_iterator]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {2, 3, 4});
    TensorIterator<float> it(t);
    REQUIRE(it.ndim() == 3);
    REQUIRE(it.numel() == 24);
    REQUIRE(it.shape() == std::vector<int64_t>({2, 3, 4}));
    REQUIRE(it.strides() == std::vector<int64_t>({12, 4, 1}));
}

TEST_CASE("TensorIterator contiguous fast path returns correct elements", "[tensor_iterator]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {2, 3});
    float vals[] = {10, 20, 30, 40, 50, 60};
    std::memcpy(t.data<float>(), vals, 6 * sizeof(float));

    TensorIterator<float> it(t);
    REQUIRE(it.is_contiguous());
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(it[i] == Catch::Approx(vals[i]));
    }
}

TEST_CASE("TensorIterator contiguous fast path write-then-read", "[tensor_iterator]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {4});
    TensorIterator<float> it(t);
    REQUIRE(it.is_contiguous());

    it[0] = 1.5f;
    it[1] = 2.5f;
    it[2] = 3.5f;
    it[3] = 4.5f;

    REQUIRE(it[0] == Catch::Approx(1.5f));
    REQUIRE(it[1] == Catch::Approx(2.5f));
    REQUIRE(it[2] == Catch::Approx(3.5f));
    REQUIRE(it[3] == Catch::Approx(4.5f));
}

TEST_CASE("TensorIterator strided path returns correct elements for transposed tensor", "[tensor_iterator]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    float vals[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(x.data<float>(), vals, 6 * sizeof(float));

    auto t = *rt.transpose(x, 0, 1);
    TensorIterator<float> it(t);
    REQUIRE_FALSE(it.is_contiguous());
    REQUIRE(it.shape() == std::vector<int64_t>({3, 2}));
    REQUIRE(it.strides() == std::vector<int64_t>({1, 3}));

    REQUIRE(it[0] == Catch::Approx(1.0f));
    REQUIRE(it[1] == Catch::Approx(4.0f));
    REQUIRE(it[2] == Catch::Approx(2.0f));
    REQUIRE(it[3] == Catch::Approx(5.0f));
    REQUIRE(it[4] == Catch::Approx(3.0f));
    REQUIRE(it[5] == Catch::Approx(6.0f));
}

TEST_CASE("TensorIterator handles non-zero storage_offset on contiguous tensor", "[tensor_iterator]") {
    Runtime rt;
    auto storage = rt.allocator().allocate(TensorMetadata::contiguous({8}, DType::Float32));
    auto* raw = static_cast<float*>(storage->data);
    for (int i = 0; i < 8; ++i) raw[i] = static_cast<float>(i + 100);

    TensorMetadata type({3}, {1}, DType::Float32);
    Tensor t(type, storage, false, 2);

    TensorIterator<float> it(t);
    REQUIRE(it.is_contiguous());
    REQUIRE(it[0] == Catch::Approx(102.0f));
    REQUIRE(it[1] == Catch::Approx(103.0f));
    REQUIRE(it[2] == Catch::Approx(104.0f));
}

TEST_CASE("TensorIterator handles non-zero storage_offset on transposed tensor", "[tensor_iterator]") {
    Runtime rt;
    auto storage = rt.allocator().allocate(TensorMetadata::contiguous({12}, DType::Float32));
    auto* raw = static_cast<float*>(storage->data);
    for (int i = 0; i < 12; ++i) raw[i] = static_cast<float>(i + 1);

    TensorMetadata t_type({3, 2}, {1, 3}, DType::Float32);
    Tensor t(t_type, storage, false, 6);

    TensorIterator<float> it(t);
    REQUIRE_FALSE(it.is_contiguous());

    REQUIRE(it[0] == Catch::Approx(7.0f));
    REQUIRE(it[1] == Catch::Approx(10.0f));
    REQUIRE(it[2] == Catch::Approx(8.0f));
    REQUIRE(it[3] == Catch::Approx(11.0f));
    REQUIRE(it[4] == Catch::Approx(9.0f));
    REQUIRE(it[5] == Catch::Approx(12.0f));
}

TEST_CASE("TensorIterator with one-dim tensor is contiguous", "[tensor_iterator]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {5});
    float vals[] = {1, 2, 3, 4, 5};
    std::memcpy(t.data<float>(), vals, 5 * sizeof(float));

    TensorIterator<float> it(t);
    REQUIRE(it.is_contiguous());
    REQUIRE(it.ndim() == 1);
    REQUIRE(it.numel() == 5);
    for (int64_t i = 0; i < 5; ++i) {
        REQUIRE(it[i] == Catch::Approx(vals[i]));
    }
}

TEST_CASE("TensorIterator const iterator returns correct values and is read-only", "[tensor_iterator]") {
    Runtime rt;
    auto t = Tensor::empty(rt, {2, 3});
    float vals[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(t.data<float>(), vals, 6 * sizeof(float));

    const TensorIterator<float> it(t);
    REQUIRE(it[0] == Catch::Approx(1.0f));
    REQUIRE(it[5] == Catch::Approx(6.0f));
}

TEST_CASE("TensorIterator strided write modifies underlying storage", "[tensor_iterator]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    float vals[] = {0, 0, 0, 0, 0, 0};
    std::memcpy(x.data<float>(), vals, 6 * sizeof(float));

    auto t = *rt.transpose(x, 0, 1);
    TensorIterator<float> it(t);

    it[0] = 10.0f;
    it[3] = 20.0f;

    REQUIRE(x.data<float>()[0] == Catch::Approx(10.0f));
    REQUIRE(x.data<float>()[4] == Catch::Approx(20.0f));
}
