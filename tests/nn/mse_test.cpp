#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/mse.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("MSELossOp forward returns scalar loss", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::zeros(rt, {3, 5});
    pred.set_requires_grad(true);
    auto target = Tensor::zeros(rt, {3, 5});

    auto loss = MSELossOp::forward(rt, pred, target);
    REQUIRE(loss);
    REQUIRE(loss.value().type().shape() == std::vector<int64_t>({1}));
}

TEST_CASE("MSELossOp zero loss when pred == target", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::zeros(rt, {2, 3});
    pred.set_requires_grad(true);
    for (int i = 0; i < 6; ++i) pred.data<float>()[i] = 2.0f;
    auto target = Tensor::zeros(rt, {2, 3});
    for (int i = 0; i < 6; ++i) target.data<float>()[i] = 2.0f;

    auto loss = MSELossOp::forward(rt, pred, target);
    REQUIRE(loss);
    REQUIRE(loss.value().data<float>()[0] == Catch::Approx(0.0f));
}

TEST_CASE("MSELossOp non-zero loss", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::zeros(rt, {1, 2});
    pred.set_requires_grad(true);
    pred.data<float>()[0] = 1.0f; pred.data<float>()[1] = 2.0f;
    auto target = Tensor::zeros(rt, {1, 2});
    target.data<float>()[0] = 3.0f; target.data<float>()[1] = 4.0f;

    auto loss = MSELossOp::forward(rt, pred, target);
    REQUIRE(loss);
    REQUIRE(loss.value().data<float>()[0] == Catch::Approx(4.0f));  // ((1-3)^2 + (2-4)^2) / 2 = (4+4)/2 = 4
}
