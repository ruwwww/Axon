#include <catch2/catch_test_macros.hpp>
#include "axon/nn/resnet18.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("BasicBlock same-dimension forward produces correct shape", "[resnet][basicblock]") {
    Runtime rt;
    BasicBlock block(rt, 64, 64, 1);
    auto x = Tensor::randn(rt, {2, 64, 56, 56});
    auto y = block.forward(rt, x);
    REQUIRE(y);
    REQUIRE((*y).type().shape() == std::vector<int64_t>({2, 64, 56, 56}));
}

TEST_CASE("BasicBlock stride-2 forward produces halved spatial dims", "[resnet][basicblock]") {
    Runtime rt;
    BasicBlock block(rt, 64, 128, 2);
    auto x = Tensor::randn(rt, {2, 64, 56, 56});
    auto y = block.forward(rt, x);
    REQUIRE(y);
    REQUIRE((*y).type().shape() == std::vector<int64_t>({2, 128, 28, 28}));
}

TEST_CASE("ResNet18 forward produces (N, 1000) output", "[resnet]") {
    Runtime rt;
    ResNet18 resnet(rt);
    auto x = Tensor::randn(rt, {2, 3, 224, 224});
    auto y = resnet.forward(rt, x);
    REQUIRE(y);
    REQUIRE((*y).type().shape() == std::vector<int64_t>({2, 1000}));
}

TEST_CASE("ResNet18 parameters are accessible and trainable", "[resnet]") {
    Runtime rt;
    ResNet18 resnet(rt);
    auto params = resnet.parameters();
    REQUIRE(params.size() > 0);
    for (auto* p : params) {
        REQUIRE(p->trainable());
        REQUIRE(p->tensor().defined());
    }
}
