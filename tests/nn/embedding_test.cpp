#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/embedding.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("Embedding creates correct weight shape", "[nn][embedding]") {
    Runtime rt;
    Embedding emb(rt, 10, 3);

    auto params = emb.parameters();
    REQUIRE(params.size() == 1);
    REQUIRE(params[0]->tensor().type().shape() == std::vector<int64_t>({10, 3}));
}

TEST_CASE("Embedding forward looks up correct indices", "[nn][embedding]") {
    Runtime rt;
    Embedding emb(rt, 5, 2);

    // Set weight to known values
    auto& w = emb.parameters()[0]->tensor();
    w.data<float>()[0] = 0.0f; w.data<float>()[1] = 0.1f;
    w.data<float>()[2] = 1.0f; w.data<float>()[3] = 1.1f;
    w.data<float>()[4] = 2.0f; w.data<float>()[5] = 2.1f;
    w.data<float>()[6] = 3.0f; w.data<float>()[7] = 3.1f;
    w.data<float>()[8] = 4.0f; w.data<float>()[9] = 4.1f;

    auto indices = Tensor::zeros(rt, {3}, DType::Int64);
    indices.data<int64_t>()[0] = 0;
    indices.data<int64_t>()[1] = 2;
    indices.data<int64_t>()[2] = 4;

    auto result = emb.forward(rt, indices);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 2}));

    REQUIRE(result.value().data<float>()[0] == Catch::Approx(0.0f));
    REQUIRE(result.value().data<float>()[1] == Catch::Approx(0.1f));
    REQUIRE(result.value().data<float>()[2] == Catch::Approx(2.0f));
    REQUIRE(result.value().data<float>()[3] == Catch::Approx(2.1f));
    REQUIRE(result.value().data<float>()[4] == Catch::Approx(4.0f));
    REQUIRE(result.value().data<float>()[5] == Catch::Approx(4.1f));
}
