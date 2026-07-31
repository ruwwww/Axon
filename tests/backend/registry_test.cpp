#include <catch2/catch_test_macros.hpp>
#include "axon/backend/registry.h"

static bool g_scalar_called = false;
static bool g_avx2_called = false;
static bool g_kernel_key_called = false;

static axon::Expected<void> mock_add_scalar(axon::cpu::KernelContext& ctx) {
    g_scalar_called = true;
    return {};
}

static axon::Expected<void> mock_add_avx2(axon::cpu::KernelContext& ctx) {
    g_avx2_called = true;
    return {};
}

static axon::Expected<void> mock_key_kernel(axon::cpu::KernelContext& ctx) {
    g_kernel_key_called = true;
    return {};
}

TEST_CASE("KernelRegistry registers and dispatches kernels based on CPU ISA", "[backend][registry]") {
    auto& reg = axon::cpu::KernelRegistry::instance();

    reg.register_kernel("mock_add", axon::cpu::ISA::Scalar, mock_add_scalar);
    reg.register_kernel("mock_add", axon::cpu::ISA::AVX2, mock_add_avx2);

    g_scalar_called = false;
    g_avx2_called = false;

    auto fn = reg.dispatch("mock_add");
    REQUIRE(fn != nullptr);

    axon::Tensor dummy_out, dummy_a, dummy_b;
    axon::Tensor outputs[] = {dummy_out};
    axon::Tensor inputs[] = {dummy_a, dummy_b};
    axon::cpu::KernelContext ctx{.outputs = outputs, .inputs = inputs};

    auto res = fn(ctx);
    REQUIRE(res);

    if (axon::cpu::has_avx2()) {
        REQUIRE(g_avx2_called);
        REQUIRE_FALSE(g_scalar_called);
    } else {
        REQUIRE(g_scalar_called);
        REQUIRE_FALSE(g_avx2_called);
    }
}

TEST_CASE("KernelRegistry registers and dispatches with KernelKey", "[backend][registry]") {
    auto& reg = axon::cpu::KernelRegistry::instance();

    axon::cpu::KernelKey key{
        .op = axon::cpu::OpId::Conv2D,
        .device = axon::Device::CPU,
        .dtype = axon::DType::Float32,
        .provider = axon::cpu::Provider::AxonNative
    };

    reg.register_kernel(key, mock_key_kernel);

    g_kernel_key_called = false;
    auto fn = reg.lookup(key);
    REQUIRE(fn != nullptr);

    axon::Tensor dummy_out;
    axon::Tensor outputs[] = {dummy_out};
    axon::cpu::KernelContext ctx{.outputs = outputs};

    auto res = fn(ctx);
    REQUIRE(res);
    REQUIRE(g_kernel_key_called);
}
