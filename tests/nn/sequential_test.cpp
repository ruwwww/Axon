#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/sequential.h"
#include "axon/nn/flatten.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Sequential chains modules correctly", "[nn][sequential]") {
    Runtime rt;
    auto seq = std::make_unique<Sequential>();
    seq->add(std::make_unique<Flatten>());

    auto x = Tensor::randn(rt, {2, 3, 4, 5});

    auto result = seq->forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 60}));
}

TEST_CASE("Sequential with multiple modules", "[nn][sequential]") {
    Runtime rt;
    auto seq = std::make_unique<Sequential>();
    seq->add(std::make_unique<Flatten>());

    // Test that two flattens in sequence work
    auto x = Tensor::randn(rt, {2, 3, 4});

    auto result = seq->forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 12}));
}
