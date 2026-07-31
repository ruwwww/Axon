#include <catch2/catch_test_macros.hpp>
#include "axon/data/dataset.h"
#include "axon/data/mnist.h"
#include "axon/data/dataloader.h"
#include "axon/runtime/runtime.h"

using namespace axon;

struct FakeDataset : Dataset {
    size_t count;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> target_shape;

    FakeDataset(size_t n, std::vector<int64_t> in_shape, std::vector<int64_t> targ_shape)
        : count(n), input_shape(std::move(in_shape)), target_shape(std::move(targ_shape)) {}

    size_t size() const override { return count; }

    std::pair<Tensor, Tensor> get(size_t index) override {
        Tensor input(TensorMetadata::contiguous(input_shape, DType::Float32), std::make_shared<Storage>(input_shape[0] * 4), false);
        Tensor target(TensorMetadata::contiguous(target_shape, DType::Int64), std::make_shared<Storage>(target_shape[0] * 8), false);
        input.data<float>()[0] = static_cast<float>(index);
        target.data<int64_t>()[0] = static_cast<int64_t>(index % 10);
        return {input, target};
    }
};

TEST_CASE("Dataset size method", "[data][dataset]") {
    FakeDataset ds(100, {784}, {1});
    REQUIRE(ds.size() == 100);
}

TEST_CASE("MNIST constructor fails on non-existent path", "[data][mnist]") {
    Runtime rt;
    bool threw = false;
    try {
        MNIST mnist(rt, "nonexistent/path", true);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE("DataLoader produces correct number of batches", "[data][dataloader]") {
    FakeDataset ds(100, {784}, {1});
    DataLoader loader(ds, 32, false);
    auto batches = loader.iter();

    REQUIRE(batches.size() == 4);
    REQUIRE(batches[0].inputs.type().shape() == std::vector<int64_t>({32, 784}));
    REQUIRE(batches[0].targets.type().shape() == std::vector<int64_t>({32, 1}));
    REQUIRE(batches[3].inputs.type().shape() == std::vector<int64_t>({4, 784}));
}

TEST_CASE("DataLoader shuffles when enabled", "[data][dataloader]") {
    FakeDataset ds(10, {1}, {1});
    DataLoader loader(ds, 5, true);
    auto batches = loader.iter();
    REQUIRE(batches.size() == 2);
}

TEST_CASE("DataLoader respects batch_size of 1", "[data][dataloader]") {
    FakeDataset ds(5, {10}, {1});
    DataLoader loader(ds, 1, false);
    auto batches = loader.iter();
    REQUIRE(batches.size() == 5);
    for (auto& b : batches) {
        REQUIRE(b.inputs.type().shape() == std::vector<int64_t>({1, 10}));
    }
}
