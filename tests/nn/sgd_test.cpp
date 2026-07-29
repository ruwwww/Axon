#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/sgd.h"
#include "axon/nn/parameter.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("SGD zero_grad clears parameter gradients", "[nn][optimizer]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {2, 2});
    t.set_requires_grad(true);
    Parameter p(t, true);
    // Manually set grad
    auto& g = p.grad();
    g.data<float>()[0] = 5.0f;

    std::vector<Parameter*> params = {&p};
    SGD opt(rt, params, 0.01f);
    opt.zero_grad();

    REQUIRE(g.data<float>()[0] == Catch::Approx(0.0f));
}

TEST_CASE("SGD step applies correct update without momentum", "[nn][optimizer]") {
    Runtime rt;
    auto t = Tensor::zeros(rt, {1, 3});
    t.set_requires_grad(true);
    t.data<float>()[0] = 1.0f; t.data<float>()[1] = 2.0f; t.data<float>()[2] = 3.0f;

    Parameter p(t, true);
    auto& g = p.grad();
    g.data<float>()[0] = 0.1f; g.data<float>()[1] = 0.2f; g.data<float>()[2] = 0.3f;

    std::vector<Parameter*> params = {&p};
    SGD opt(rt, params, 0.5f);
    opt.step();

    REQUIRE(t.data<float>()[0] == Catch::Approx(0.95f));
    REQUIRE(t.data<float>()[1] == Catch::Approx(1.9f));
    REQUIRE(t.data<float>()[2] == Catch::Approx(2.85f));
}
