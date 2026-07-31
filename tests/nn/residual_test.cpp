#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/nn/residual.h"
#include "axon/nn/flatten.h"
#include "axon/nn/module.h"
#include "axon/runtime/runtime.h"

using namespace axon;

struct IdentityModule : Module {
    Expected<Tensor> forward(Runtime& rt, const Tensor& x) override {
        auto out_type = TensorMetadata::contiguous(x.type().shape(), x.type().dtype());
        auto out = Tensor(out_type, rt.allocator().allocate(out_type), false);
        auto* o_ptr = out.data<float>();
        auto* x_ptr = x.data<const float>();
        auto n = x.type().numel();
        for (int64_t i = 0; i < n; ++i) o_ptr[i] = x_ptr[i];
        return out;
    }
};

TEST_CASE("Residual forward adds skip connection", "[nn][residual]") {
    Runtime rt;
    auto inner = std::make_unique<IdentityModule>();
    Residual res(std::move(inner));

    auto x = Tensor::ones(rt, {2, 3});

    auto result = res.forward(rt, x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 3}));
    // Identity(x) = x, so residual = x + x = 2x
    auto* data = result.value().data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(2.0f));
    }
}
