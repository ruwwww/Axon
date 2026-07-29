#include <catch2/catch_test_macros.hpp>
#include "axon/nn/module.h"
#include "axon/nn/parameter.h"
#include "axon/runtime/runtime.h"

using namespace axon;

struct TestModule : Module {
    Parameter w;

    TestModule(Runtime& rt) : w(Tensor::zeros(rt, {2, 3}), true) {
        register_parameter("weight", &w);
    }

    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override {
        return Tensor::zeros(rt, {2, 2});
    }
};

TEST_CASE("Module registers and returns parameters", "[nn][module]") {
    Runtime rt;
    TestModule m(rt);

    auto params = m.parameters();
    REQUIRE(params.size() == 1);
    REQUIRE(params[0]->tensor().type().shape() == std::vector<int64_t>({2, 3}));
}

TEST_CASE("Module train/eval toggles mode", "[nn][module]") {
    Runtime rt;
    TestModule m(rt);

    REQUIRE(m.is_training());
    m.eval();
    REQUIRE_FALSE(m.is_training());
    m.train();
    REQUIRE(m.is_training());
}

TEST_CASE("Module with multiple parameters", "[nn][module]") {
    Runtime rt;
    Parameter w1(Tensor::zeros(rt, {2, 3}), true);
    Parameter w2(Tensor::zeros(rt, {4, 5}), true);

    TestModule m(rt);
    m.register_parameter("w2", &w2);

    auto params = m.parameters();
    REQUIRE(params.size() == 2);
}

TEST_CASE("Module forward is callable", "[nn][module]") {
    Runtime rt;
    TestModule m(rt);
    auto x = Tensor::zeros(rt, {2, 3});

    auto result = m.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 2}));
}
