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

// ── Conv2d ─────────────────────────────────────────────────────────────

TEST_CASE("cpu::conv2d basic 1x1 kernel produces correct output", "[backend][cpu]") {
    Runtime rt;
    auto input = rt.empty({1, 1, 3, 3});
    float inp_data[] = {1,2,3,4,5,6,7,8,9};
    memcpy(input.data<float>(), inp_data, 9 * sizeof(float));

    auto weight = rt.empty({1, 1, 1, 1});
    weight.data<float>()[0] = 2.0f;

    auto out = rt.empty({1, 1, 3, 3});
    auto result = cpu::conv2d(out, input, weight, 1, 0);
    REQUIRE(result);

    for (int64_t i = 0; i < 9; ++i) {
        REQUIRE(out.data<float>()[i] == Catch::Approx(inp_data[i] * 2.0f));
    }
}

TEST_CASE("cpu::conv2d 3x3 kernel with stride 2 and padding 0", "[backend][cpu]") {
    Runtime rt;
    // input: N=1, C=1, H=5, W=5
    auto input = rt.empty({1, 1, 5, 5});
    float val = 0.0f;
    for (int64_t i = 0; i < 25; ++i) input.data<float>()[i] = val++;

    // weight: OC=1, IC=1, KH=3, KW=3 (all ones)
    auto weight = rt.empty({1, 1, 3, 3});
    for (int64_t i = 0; i < 9; ++i) weight.data<float>()[i] = 1.0f;

    // Out shape: ceil((5 - 3) / 2) + 1 = 2
    auto out = rt.empty({1, 1, 2, 2});
    auto result = cpu::conv2d(out, input, weight, 2, 0);
    REQUIRE(result);

    REQUIRE(out.data<float>()[0] == Catch::Approx(54.0f));   // (0,0): rows 0-2, cols 0-2
    REQUIRE(out.data<float>()[1] == Catch::Approx(72.0f));   // (0,1): rows 0-2, cols 2-4
    REQUIRE(out.data<float>()[2] == Catch::Approx(144.0f));  // (1,0): rows 2-4, cols 0-2
    REQUIRE(out.data<float>()[3] == Catch::Approx(162.0f));  // (1,1): rows 2-4, cols 2-4
}

TEST_CASE("cpu::conv2d with padding produces correct output shape", "[backend][cpu]") {
    Runtime rt;
    auto input = rt.ones({1, 1, 4, 4});
    auto weight = rt.ones({1, 1, 3, 3});
    // padding=1, stride=1 => output 4x4
    auto out = rt.empty({1, 1, 4, 4});
    auto result = cpu::conv2d(out, input, weight, 1, 1);
    REQUIRE(result);
    // Each element is sum of 3x3 area of ones (except edges, but padded with zeros)
    // Corner: 4 ones, Edge: 6 ones, Center: 9 ones
    REQUIRE(out.data<float>()[0] == Catch::Approx(4.0f));
    REQUIRE(out.data<float>()[5] == Catch::Approx(9.0f));  // center (1,1): all 9 ones
}

// ── MaxPool2d ──────────────────────────────────────────────────────────

TEST_CASE("cpu::maxpool2d basic 2x2 kernel with stride 2", "[backend][cpu]") {
    Runtime rt;
    auto input = rt.empty({1, 1, 4, 4});
    float inp_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    memcpy(input.data<float>(), inp_data, 16 * sizeof(float));

    auto out = rt.empty({1, 1, 2, 2});
    auto result = cpu::maxpool2d(out, input, 2, 2);
    REQUIRE(result);

    REQUIRE(out.data<float>()[0] == Catch::Approx(6.0f));
    REQUIRE(out.data<float>()[1] == Catch::Approx(8.0f));
    REQUIRE(out.data<float>()[2] == Catch::Approx(14.0f));
    REQUIRE(out.data<float>()[3] == Catch::Approx(16.0f));
}

TEST_CASE("cpu::maxpool2d with non-square input", "[backend][cpu]") {
    Runtime rt;
    // input: N=1, C=1, H=3, W=5
    auto input = rt.empty({1, 1, 3, 5});
    for (int64_t i = 0; i < 15; ++i) input.data<float>()[i] = static_cast<float>(i);

    auto out = rt.empty({1, 1, 1, 1});
    auto result = cpu::maxpool2d(out, input, 3, 3);
    REQUIRE(result);

    // OW = floor((5-3)/3)+1 = 1, single window at (0,0): rows 0-2, cols 0-2 = max(0..2,5..7,10..12) = 12
    REQUIRE(out.type().shape() == std::vector<int64_t>({1, 1, 1, 1}));
    REQUIRE(out.data<float>()[0] == Catch::Approx(12.0f));
}

// ── BatchNorm ──────────────────────────────────────────────────────────

TEST_CASE("cpu::batchnorm during training produces zero mean output", "[backend][cpu]") {
    Runtime rt;
    // N=2, C=3, H=2, W=2 => shape {2,3,2,2}
    auto input = rt.empty({2, 3, 2, 2});
    float inp_data[] = {
        1,2, 3,4,  5,6, 7,8,  9,10, 11,12,
        13,14, 15,16,  17,18, 19,20,  21,22, 23,24
    };
    memcpy(input.data<float>(), inp_data, 24 * sizeof(float));

    auto gamma = rt.ones({3});
    auto beta = rt.zeros({3});
    auto running_mean = rt.zeros({3});
    auto running_var = rt.ones({3});

    auto out = rt.empty({2, 3, 2, 2});
    float momentum = 0.9f;
    float epsilon = 1e-5f;
    auto result = cpu::batchnorm(out, input, gamma, beta, running_mean, running_var, momentum, epsilon, true);
    REQUIRE(result);

    // After batch norm, each channel should have mean ~0, var ~1
    float* out_ptr = out.data<float>();
    for (int64_t c = 0; c < 3; ++c) {
        float sum = 0.0f;
        for (int64_t n = 0; n < 2; ++n) {
            for (int64_t hw = 0; hw < 4; ++hw) {
                sum += out_ptr[n * 3 * 4 + c * 4 + hw];
            }
        }
        float mean = sum / 8.0f;
        REQUIRE(std::abs(mean) < 1e-5f);
    }
}

TEST_CASE("cpu::batchnorm in eval uses running stats", "[backend][cpu]") {
    Runtime rt;
    auto input = rt.ones({1, 2, 1, 1});
    auto gamma = rt.ones({2});
    auto beta = rt.zeros({2});
    auto running_mean = rt.zeros({2});
    auto running_var = rt.ones({2});

    auto out = rt.empty({1, 2, 1, 1});
    auto result = cpu::batchnorm(out, input, gamma, beta, running_mean, running_var, 0.9f, 1e-5f, false);
    REQUIRE(result);

    // In eval mode with running_mean=0, running_var=1: output = (input - 0) / sqrt(1 + eps) * gamma + beta
    REQUIRE(out.data<float>()[0] == Catch::Approx(1.0f));
    REQUIRE(out.data<float>()[1] == Catch::Approx(1.0f));
}

// ── LayerNorm ──────────────────────────────────────────────────────────

TEST_CASE("cpu::layernorm normalizes over last dimension", "[backend][cpu]") {
    Runtime rt;
    // N=2, C=4
    auto input = rt.empty({2, 4});
    float inp_data[] = {1,2,3,4, 5,6,7,8};
    memcpy(input.data<float>(), inp_data, 8 * sizeof(float));

    auto gamma = rt.ones({4});
    auto beta = rt.zeros({4});

    auto out = rt.empty({2, 4});
    auto result = cpu::layernorm(out, input, gamma, beta, 1e-5f);
    REQUIRE(result);

    // Each row should have mean ~0, var ~1
    for (int64_t n = 0; n < 2; ++n) {
        float sum = 0.0f;
        for (int64_t c = 0; c < 4; ++c) sum += out.data<float>()[n * 4 + c];
        float mean = sum / 4.0f;
        REQUIRE(std::abs(mean) < 1e-5f);

        float var_sum = 0.0f;
        for (int64_t c = 0; c < 4; ++c) {
            float diff = out.data<float>()[n * 4 + c] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / 4.0f;
        REQUIRE(std::abs(var - 1.0f) < 1e-4f);
    }
}

TEST_CASE("cpu::layernorm with affine parameters shifts output", "[backend][cpu]") {
    Runtime rt;
    auto input = rt.zeros({1, 3});
    auto gamma = rt.empty({3});
    gamma.data<float>()[0] = 1.0f; gamma.data<float>()[1] = 2.0f; gamma.data<float>()[2] = 3.0f;
    auto beta = rt.empty({3});
    beta.data<float>()[0] = 0.1f; beta.data<float>()[1] = 0.2f; beta.data<float>()[2] = 0.3f;

    auto out = rt.empty({1, 3});
    auto result = cpu::layernorm(out, input, gamma, beta, 1e-5f);
    REQUIRE(result);

    // Input is all zeros: normalized gives zeros, then affine: out = gamma*0 + beta = beta
    REQUIRE(out.data<float>()[0] == Catch::Approx(0.1f));
    REQUIRE(out.data<float>()[1] == Catch::Approx(0.2f));
    REQUIRE(out.data<float>()[2] == Catch::Approx(0.3f));
}
