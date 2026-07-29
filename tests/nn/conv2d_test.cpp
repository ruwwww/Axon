#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include "axon/nn/conv2d.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Conv2D creates correct weight shape", "[nn][conv2d]") {
    Runtime rt;
    Conv2D conv(rt, 3, 6, 3);

    auto params = conv.parameters();
    REQUIRE(params.size() == 2);

    auto& w = *params[0];
    REQUIRE(w.tensor().type().shape() == std::vector<int64_t>({6, 3, 3, 3}));
}

TEST_CASE("Conv2D forward produces correct output shape", "[nn][conv2d]") {
    Runtime rt;
    Conv2D conv(rt, 3, 6, 3, 1, 1);
    auto x = Tensor::zeros(rt, {2, 3, 32, 32});

    auto result = conv.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 6, 32, 32}));
}

TEST_CASE("Conv2D forward with stride 2", "[nn][conv2d]") {
    Runtime rt;
    Conv2D conv(rt, 1, 2, 3, 2, 0);
    auto x = Tensor::zeros(rt, {1, 1, 7, 7});

    auto result = conv.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({1, 2, 3, 3}));
}

TEST_CASE("Conv2D without bias", "[nn][conv2d]") {
    Runtime rt;
    Conv2D conv(rt, 3, 6, 3, 1, 1, false);

    auto params = conv.parameters();
    REQUIRE(params.size() == 1);
}
