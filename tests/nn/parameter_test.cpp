#include <catch2/catch_test_macros.hpp>
#include "axon/nn/parameter.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Parameter wraps a tensor", "[nn][parameter]") {
    Runtime rt;
    Tensor t = Tensor::zeros(rt, {3, 4});
    Parameter p(t, true);

    REQUIRE(p.tensor().defined());
    REQUIRE(p.tensor().type().shape() == std::vector<int64_t>({3, 4}));
    REQUIRE(p.trainable());
}

TEST_CASE("Parameter gradient is allocated lazily", "[nn][parameter]") {
    Runtime rt;
    Tensor t = Tensor::zeros(rt, {2, 2});
    Parameter p(t, true);

    REQUIRE_FALSE(p.has_grad());

    auto& g = p.grad();
    REQUIRE(p.has_grad());
    REQUIRE(g.type().shape() == std::vector<int64_t>({2, 2}));
}

TEST_CASE("Parameter can be non-trainable", "[nn][parameter]") {
    Runtime rt;
    Tensor t = Tensor::zeros(rt, {3, 3});
    Parameter p(t, false);

    REQUIRE_FALSE(p.trainable());
}
