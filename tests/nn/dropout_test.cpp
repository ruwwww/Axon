#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/dropout.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Dropout eval mode is no-op", "[nn][dropout]") {
    Runtime rt;
    Dropout dropout(0.5f);
    dropout.eval();
    auto x = Tensor::ones(rt, {2, 3});

    auto result = dropout.forward(rt, x);
    REQUIRE(result);
    auto* data = result.value().data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(1.0f));
    }
}

TEST_CASE("Dropout with p=0 is no-op", "[nn][dropout]") {
    Runtime rt;
    Dropout dropout(0.0f);
    auto x = Tensor::ones(rt, {2, 3});

    auto result = dropout.forward(rt, x);
    REQUIRE(result);
    auto* data = result.value().data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(1.0f));
    }
}

TEST_CASE("Dropout forward preserves shape", "[nn][dropout]") {
    Runtime rt;
    Dropout dropout(0.5f);
    auto x = Tensor::randn(rt, {3, 4, 5});

    auto result = dropout.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 4, 5}));
}
