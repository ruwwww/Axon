#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include "axon/nn/linear.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Linear creates correct weight shape", "[nn][linear]") {
    Runtime rt;
    Linear linear(rt, 10, 5);

    auto params = linear.parameters();
    REQUIRE(params.size() == 2);

    auto& w = *params[0];
    REQUIRE(w.trainable());
    REQUIRE(w.tensor().type().shape() == std::vector<int64_t>({10, 5}));  // in x out
}

TEST_CASE("Linear forward produces correct output shape", "[nn][linear]") {
    Runtime rt;
    Linear linear(rt, 10, 5);
    auto x = Tensor::zeros(rt, {3, 10});

    auto result = linear.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 5}));
}

TEST_CASE("Linear forward computes correct values", "[nn][linear]") {
    Runtime rt;
    Linear linear(rt, 3, 2);
    auto x = Tensor::zeros(rt, {1, 3});
    x.data<float>()[0] = 1.0f; x.data<float>()[1] = 2.0f; x.data<float>()[2] = 3.0f;

    auto params = linear.parameters();
    auto& w_tensor = params[0]->tensor();
    auto& b_tensor = params[1]->tensor();

    w_tensor.data<float>()[0] = 1.0f; w_tensor.data<float>()[1] = 0.0f;
    w_tensor.data<float>()[2] = 0.0f; w_tensor.data<float>()[3] = 1.0f;
    w_tensor.data<float>()[4] = 0.0f; w_tensor.data<float>()[5] = 0.0f;

    b_tensor.data<float>()[0] = 0.1f; b_tensor.data<float>()[1] = 0.2f;

    auto result = linear.forward(rt, x);
    REQUIRE(result);

    REQUIRE(result.value().data<float>()[0] == Catch::Approx(1.1f));
    REQUIRE(result.value().data<float>()[1] == Catch::Approx(2.2f));
}

TEST_CASE("Linear without bias", "[nn][linear]") {
    Runtime rt;
    Linear linear(rt, 10, 5, false);

    auto params = linear.parameters();
    REQUIRE(params.size() == 1);

    auto x = Tensor::zeros(rt, {3, 10});
    auto result = linear.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 5}));
}
