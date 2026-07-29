#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/l1_loss.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("L1LossOp forward returns scalar loss", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::zeros(rt, {3, 5});
    pred.set_requires_grad(true);
    auto target = Tensor::zeros(rt, {3, 5});

    auto loss = L1LossOp::forward(rt, pred, target);
    REQUIRE(loss);
    REQUIRE(loss.value().type().shape() == std::vector<int64_t>({1}));
}

TEST_CASE("L1LossOp zero loss when pred == target", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::zeros(rt, {2, 3});
    pred.set_requires_grad(true);
    for (int i = 0; i < 6; ++i) pred.data<float>()[i] = 2.0f;
    auto target = Tensor::zeros(rt, {2, 3});
    for (int i = 0; i < 6; ++i) target.data<float>()[i] = 2.0f;

    auto loss = L1LossOp::forward(rt, pred, target);
    REQUIRE(loss);
    REQUIRE(loss.value().data<float>()[0] == Catch::Approx(0.0f));
}

TEST_CASE("L1LossOp non-zero loss", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::zeros(rt, {1, 2});
    pred.set_requires_grad(true);
    pred.data<float>()[0] = 1.0f; pred.data<float>()[1] = 2.0f;
    auto target = Tensor::zeros(rt, {1, 2});
    target.data<float>()[0] = 3.0f; target.data<float>()[1] = 4.0f;

    auto loss = L1LossOp::forward(rt, pred, target);
    REQUIRE(loss);
    // (|1-3| + |2-4|) / 2 = (2 + 2) / 2 = 2
    REQUIRE(loss.value().data<float>()[0] == Catch::Approx(2.0f));
}

TEST_CASE("L1LossOp backward matches finite differences", "[nn][loss]") {
    Runtime rt;
    auto pred = Tensor::empty(rt, {4});
    pred.set_requires_grad(true);
    auto target = Tensor::empty(rt, {4});

    float p_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float t_data[] = {2.0f, 1.0f, 5.0f, 3.0f};
    std::memcpy(pred.data<float>(), p_data, 4 * sizeof(float));
    std::memcpy(target.data<float>(), t_data, 4 * sizeof(float));

    Tensor loss = *L1LossOp::forward(rt, pred, target);
    auto result = rt.autograd().backward(rt, loss);
    REQUIRE(result);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(pred.id()) > 0);
    Tensor grad_pred = grads[pred.id()];
    REQUIRE(grad_pred.type().shape() == std::vector<int64_t>({4}));

    const double eps = 1.0 / 1024.0;
    for (int64_t i = 0; i < 4; ++i) {
        double orig = p_data[i];

        pred.data<float>()[i] = static_cast<float>(orig + eps);
        double l_plus = 0.0;
        {
            auto* p = pred.data<const float>();
            auto* t = target.data<const float>();
            for (int64_t j = 0; j < 4; ++j) {
                l_plus += std::abs(static_cast<double>(p[j]) - static_cast<double>(t[j]));
            }
            l_plus /= 4.0;
        }

        pred.data<float>()[i] = static_cast<float>(orig - eps);
        double l_minus = 0.0;
        {
            auto* p = pred.data<const float>();
            auto* t = target.data<const float>();
            for (int64_t j = 0; j < 4; ++j) {
                l_minus += std::abs(static_cast<double>(p[j]) - static_cast<double>(t[j]));
            }
            l_minus /= 4.0;
        }

        pred.data<float>()[i] = static_cast<float>(orig);

        double numerical = (l_plus - l_minus) / (2.0 * eps);
        REQUIRE(static_cast<double>(grad_pred.data<float>()[i]) == Catch::Approx(numerical).epsilon(1e-3));
    }
}
