#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include "axon/nn/adamw.h"
#include "axon/nn/parameter.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("AdamW zero_grad clears parameter gradients", "[nn][optimizer]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {2, 2});
    t.set_requires_grad(true);
    Parameter p(t, true);
    auto& g = p.grad();
    g.data<float>()[0] = 5.0f;

    std::vector<Parameter*> params = {&p};
    AdamW opt(rt, params, 0.01f);
    opt.zero_grad();

    REQUIRE(g.data<float>()[0] == Catch::Approx(0.0f));
}

TEST_CASE("AdamW step applies correct update for first step", "[nn][optimizer]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {1, 2});
    t.set_requires_grad(true);
    t.data<float>()[0] = 1.0f; t.data<float>()[1] = 2.0f;

    Parameter p(t, true);
    auto& g = p.grad();
    g.data<float>()[0] = 0.1f; g.data<float>()[1] = 0.2f;

    std::vector<Parameter*> params = {&p};
    AdamW opt(rt, params, 1.0f, 0.9f, 0.999f, 1e-8f, 0.0f);
    opt.step();

    // First step: w ≈ w - g/|g| = 0 for g=0.1, 1.0 for g=0.2
    REQUIRE(std::abs(t.data<float>()[0]) < 0.01f);
    REQUIRE(std::abs(t.data<float>()[1] - 1.0f) < 0.01f);
}

TEST_CASE("AdamW weight decay is applied", "[nn][optimizer]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {1, 1});
    t.set_requires_grad(true);
    t.data<float>()[0] = 1.0f;

    Parameter p(t, true);
    p.grad().data<float>()[0] = 0.0f;

    std::vector<Parameter*> params = {&p};
    AdamW opt(rt, params, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);
    opt.step();

    REQUIRE(t.data<float>()[0] == Catch::Approx(0.95f));
}
