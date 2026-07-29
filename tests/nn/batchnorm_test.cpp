#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/batchnorm.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("BatchNorm creates correct parameter shapes", "[nn][batchnorm]") {
    Runtime rt;
    BatchNorm bn(rt, 8);

    auto params = bn.parameters();
    REQUIRE(params.size() == 2);
    REQUIRE(params[0]->tensor().type().shape() == std::vector<int64_t>({8}));
    REQUIRE(params[1]->tensor().type().shape() == std::vector<int64_t>({8}));
}

TEST_CASE("BatchNorm forward preserves shape", "[nn][batchnorm]") {
    Runtime rt;
    BatchNorm bn(rt, 4);
    auto x = Tensor::randn(rt, {3, 4, 5, 5});

    auto result = bn.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 4, 5, 5}));
}

TEST_CASE("BatchNorm eval mode uses running stats", "[nn][batchnorm]") {
    Runtime rt;
    BatchNorm bn(rt, 2);
    bn.eval();
    auto x = Tensor::ones(rt, {1, 2, 1, 1});

    auto result = bn.forward(rt, x);
    REQUIRE(result);
    auto* data = result.value().data<float>();
    REQUIRE(data[0] == Catch::Approx(1.0f));
    REQUIRE(data[1] == Catch::Approx(1.0f));
}
