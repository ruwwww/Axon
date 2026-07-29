#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/layernorm.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("LayerNorm creates correct parameter shapes", "[nn][layernorm]") {
    Runtime rt;
    LayerNorm ln(rt, {4});

    auto params = ln.parameters();
    REQUIRE(params.size() == 2);
    REQUIRE(params[0]->tensor().type().shape() == std::vector<int64_t>({4}));
}

TEST_CASE("LayerNorm forward preserves shape", "[nn][layernorm]") {
    Runtime rt;
    LayerNorm ln(rt, {8});
    auto x = Tensor::randn(rt, {3, 8});

    auto result = ln.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 8}));
}
