#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/flatten.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Flatten forward reshapes correctly", "[nn][flatten]") {
    Runtime rt;
    Flatten flatten;
    auto x = Tensor::randn(rt, {3, 4, 5, 6});

    auto result = flatten.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 120}));
}

TEST_CASE("Flatten forward preserves batch dimension", "[nn][flatten]") {
    Runtime rt;
    Flatten flatten;
    auto x = Tensor::randn(rt, {8, 2});

    auto result = flatten.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({8, 2}));
}
