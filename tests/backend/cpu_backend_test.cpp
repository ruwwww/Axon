#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "axon/backend/cpu_backend.h"
#include "axon/runtime/runtime.h"

using namespace axon;

TEST_CASE("cpu::add computes correct result", "[backend][cpu]") {
    Runtime rt;
    auto a = rt.ones({2, 3});
    auto b = rt.ones({2, 3});
    auto out = rt.empty({2, 3});

    auto result = cpu::add(out, a, b);
    REQUIRE(result);

    auto* data = out.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(2.0f));
    }
}

TEST_CASE("cpu::sub computes correct result", "[backend][cpu]") {
    Runtime rt;
    auto a = rt.ones({2, 3});
    auto b = rt.ones({2, 3});
    auto out = rt.empty({2, 3});

    auto result = cpu::sub(out, a, b);
    REQUIRE(result);

    auto* data = out.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(0.0f));
    }
}

TEST_CASE("cpu::mul computes correct result", "[backend][cpu]") {
    Runtime rt;
    auto a = rt.ones({2, 3});
    auto b = rt.zeros({2, 3});
    auto out = rt.empty({2, 3});

    auto result = cpu::mul(out, a, b);
    REQUIRE(result);

    auto* data = out.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(0.0f));
    }
}

TEST_CASE("cpu::div computes correct result", "[backend][cpu]") {
    Runtime rt;
    auto a = Tensor::ones(rt, {2, 3});
    auto b = Tensor::ones(rt, {2, 3});
    auto out = Tensor::empty(rt, {2, 3});

    {
        auto* ptr = a.data<float>();
        for (int i = 0; i < 6; ++i) ptr[i] = 6.0f;
    }
    {
        auto* ptr = b.data<float>();
        for (int i = 0; i < 6; ++i) ptr[i] = 2.0f;
    }

    auto result = cpu::div(out, a, b);
    REQUIRE(result);

    auto* data = out.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(data[i] == Catch::Approx(3.0f));
    }
}

TEST_CASE("cpu::add rejects mismatched shapes", "[backend][cpu]") {
    Runtime rt;
    auto a = rt.ones({2, 3});
    auto b = rt.ones({3, 2});
    auto out = rt.empty({2, 3});

    auto result = cpu::add(out, a, b);
    REQUIRE_FALSE(result);
}
