#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include "axon/nn/cross_entropy.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("CrossEntropyLossOp forward returns scalar loss", "[nn][loss]") {
    Runtime rt;
    auto logits = Tensor::zeros(rt, {3, 5});
    logits.set_requires_grad(true);
    auto targets = Tensor::zeros(rt, {3}, DType::Int64);

    targets.data<int64_t>()[0] = 0;
    targets.data<int64_t>()[1] = 2;
    targets.data<int64_t>()[2] = 4;

    auto loss = CrossEntropyLossOp::forward(rt, logits, targets);
    REQUIRE(loss);
    REQUIRE(loss.value().type().shape() == std::vector<int64_t>({1}));
}

TEST_CASE("CrossEntropyLossOp loss is positive for uniform logits", "[nn][loss]") {
    Runtime rt;
    auto logits = Tensor::zeros(rt, {2, 3});
    logits.set_requires_grad(true);
    // Fill with uniform values
    for (int i = 0; i < 6; ++i) logits.data<float>()[i] = 1.0f;
    auto targets = Tensor::zeros(rt, {2}, DType::Int64);
    targets.data<int64_t>()[0] = 0;
    targets.data<int64_t>()[1] = 1;

    auto loss = CrossEntropyLossOp::forward(rt, logits, targets);
    REQUIRE(loss);
    float loss_val = loss.value().data<float>()[0];
    REQUIRE(loss_val > 0.0f);
    REQUIRE(loss_val == Catch::Approx(std::log(3.0f)).epsilon(1e-5));
}
