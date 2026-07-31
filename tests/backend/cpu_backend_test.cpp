#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include "axon/backend/cpu_backend.h"
#include "axon/tensor/tensor_iterator.h"
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

// ── GELU ────────────────────────────────────────────────────────────────

TEST_CASE("cpu::gelu produces correct output for known input", "[backend][cpu]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {4});
    auto out = Tensor::empty(rt, {4});

    float x_data[] = {-2.0f, -1.0f, 0.0f, 1.0f};
    std::memcpy(x.data<float>(), x_data, 4 * sizeof(float));

    auto result = cpu::gelu(out, x);
    REQUIRE(result);

    // Expected values from GELU approximation: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    REQUIRE(out.data<float>()[0] == Catch::Approx(-0.0454f).epsilon(1e-2));
    REQUIRE(out.data<float>()[1] == Catch::Approx(-0.1588f).epsilon(1e-2));
    REQUIRE(out.data<float>()[2] == Catch::Approx(0.0f));
    REQUIRE(out.data<float>()[3] == Catch::Approx(0.8412f).epsilon(1e-2));
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

// ── reduce_mean ─────────────────────────────────────────────────────────

TEST_CASE("cpu::reduce_mean over single dim", "[backend][cpu]") {
    Runtime rt;
    auto input = Tensor::empty(rt, {2, 3});
    float data[] = {1,2,3,4,5,6};
    std::memcpy(input.data<float>(), data, 6 * sizeof(float));

    auto out = Tensor::empty(rt, {2});
    auto result = cpu::reduce_mean(out, input, {1});
    REQUIRE(result);
    REQUIRE(out.data<float>()[0] == Catch::Approx(2.0f));
    REQUIRE(out.data<float>()[1] == Catch::Approx(5.0f));
}

TEST_CASE("cpu::reduce_mean over all dims", "[backend][cpu]") {
    Runtime rt;
    auto input = Tensor::empty(rt, {2, 3});
    float data[] = {1,2,3,4,5,6};
    std::memcpy(input.data<float>(), data, 6 * sizeof(float));

    auto out = Tensor::empty(rt, {1});
    auto result = cpu::reduce_mean(out, input, {0, 1});
    REQUIRE(result);
    REQUIRE(out.data<float>()[0] == Catch::Approx(3.5f));
}

// ── Non-contiguous element-wise tests ───────────────────────────────────

TEST_CASE("cpu::add on non-contiguous (transposed) inputs", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_a = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_b = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float a_vals[] = {1,2,3,4,5,6};
    float b_vals[] = {10,20,30,40,50,60};
    std::memcpy(storage_a->data, a_vals, 6 * sizeof(float));
    std::memcpy(storage_b->data, b_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    // Transposed layout: shape {3,2}, strides {1,3}
    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor a(tt, storage_a, false, 0);
    Tensor b(tt, storage_b, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::add(out, a, b);
    REQUIRE(result);

    TensorIterator<float> it(out);
    REQUIRE(it[0] == Catch::Approx(11.0f));  // (0,0) = 1+10
    REQUIRE(it[1] == Catch::Approx(44.0f));  // (0,1) = 4+40
    REQUIRE(it[2] == Catch::Approx(22.0f));  // (1,0) = 2+20
    REQUIRE(it[3] == Catch::Approx(55.0f));  // (1,1) = 5+50
    REQUIRE(it[4] == Catch::Approx(33.0f));  // (2,0) = 3+30
    REQUIRE(it[5] == Catch::Approx(66.0f));  // (2,1) = 6+60
}

TEST_CASE("cpu::sub on non-contiguous (transposed) inputs", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_a = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_b = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float a_vals[] = {10,20,30,40,50,60};
    float b_vals[] = {1,2,3,4,5,6};
    std::memcpy(storage_a->data, a_vals, 6 * sizeof(float));
    std::memcpy(storage_b->data, b_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor a(tt, storage_a, false, 0);
    Tensor b(tt, storage_b, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::sub(out, a, b);
    REQUIRE(result);

    TensorIterator<float> it(out);
    REQUIRE(it[0] == Catch::Approx(9.0f));
    REQUIRE(it[1] == Catch::Approx(36.0f));
    REQUIRE(it[2] == Catch::Approx(18.0f));
    REQUIRE(it[3] == Catch::Approx(45.0f));
    REQUIRE(it[4] == Catch::Approx(27.0f));
    REQUIRE(it[5] == Catch::Approx(54.0f));
}

TEST_CASE("cpu::mul on non-contiguous (transposed) inputs", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_a = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_b = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float a_vals[] = {1,2,3,4,5,6};
    float b_vals[] = {2,3,4,5,6,7};
    std::memcpy(storage_a->data, a_vals, 6 * sizeof(float));
    std::memcpy(storage_b->data, b_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor a(tt, storage_a, false, 0);
    Tensor b(tt, storage_b, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::mul(out, a, b);
    REQUIRE(result);

    TensorIterator<float> it(out);
    REQUIRE(it[0] == Catch::Approx(2.0f));
    REQUIRE(it[1] == Catch::Approx(20.0f));
    REQUIRE(it[2] == Catch::Approx(6.0f));
    REQUIRE(it[3] == Catch::Approx(30.0f));
    REQUIRE(it[4] == Catch::Approx(12.0f));
    REQUIRE(it[5] == Catch::Approx(42.0f));
}

TEST_CASE("cpu::div on non-contiguous (transposed) inputs", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_a = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_b = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    // Transposed {3,2}, strides {1,3}: flat logical order = storage[0], storage[3], storage[1], storage[4], storage[2], storage[5]
    // Arrange storage so logical values are: a=[10,40,20,50,30,60], b=[2,4,5,5,10,6]
    float a_vals[] = {10, 20, 30, 40, 50, 60};   // storage
    float b_vals[] = {2, 5, 10, 4, 5, 6};
    std::memcpy(storage_a->data, a_vals, 6 * sizeof(float));
    std::memcpy(storage_b->data, b_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor a(tt, storage_a, false, 0);
    Tensor b(tt, storage_b, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::div(out, a, b);
    REQUIRE(result);

    // Logical flat order: a[0]=10,a[3]=40,a[1]=20,a[4]=50,a[2]=30,a[5]=60
    //                      b[0]=2, b[3]=4, b[1]=5, b[4]=5, b[2]=10,b[5]=6
    //                      result: 5, 10, 4, 10, 3, 10
    TensorIterator<float> it(out);
    REQUIRE(it[0] == Catch::Approx(5.0f));
    REQUIRE(it[1] == Catch::Approx(10.0f));
    REQUIRE(it[2] == Catch::Approx(4.0f));
    REQUIRE(it[3] == Catch::Approx(10.0f));
    REQUIRE(it[4] == Catch::Approx(3.0f));
    REQUIRE(it[5] == Catch::Approx(10.0f));
}

TEST_CASE("cpu::relu on non-contiguous (transposed) input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_x = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float x_vals[] = {-1, 2, -3, 4, -5, 6};
    std::memcpy(storage_x->data, x_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor x(tt, storage_x, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::relu(out, x);
    REQUIRE(result);

    TensorIterator<float> it(out);
    REQUIRE(it[0] == Catch::Approx(0.0f));
    REQUIRE(it[1] == Catch::Approx(4.0f));
    REQUIRE(it[2] == Catch::Approx(2.0f));
    REQUIRE(it[3] == Catch::Approx(0.0f));
    REQUIRE(it[4] == Catch::Approx(0.0f));
    REQUIRE(it[5] == Catch::Approx(6.0f));
}

TEST_CASE("cpu::gelu on non-contiguous (transposed) input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_x = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    // Transposed {2,2}, strides {1,2}: flat logical order = storage[0], storage[2], storage[1], storage[3]
    // Arrange storage so logical values are: [-2, -1, 0, 1]
    // storage = [-2, 0, -1, 1]
    float x_vals[] = {-2.0f, 0.0f, -1.0f, 1.0f};
    std::memcpy(storage_x->data, x_vals, 4 * sizeof(float));
    std::memset(storage_out->data, 0, 4 * sizeof(float));

    TensorType tt({2, 2}, {1, 2}, DType::Float32);
    Tensor x(tt, storage_x, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::gelu(out, x);
    REQUIRE(result);

    // Logical values: [-2, -1, 0, 1]
    // Expected gelu: gelu(-2)≈-0.0454, gelu(-1)≈-0.1588, gelu(0)=0, gelu(1)≈0.8412
    TensorIterator<float> it(out);
    REQUIRE(it[0] == Catch::Approx(-0.0454f).epsilon(1e-2));
    REQUIRE(it[1] == Catch::Approx(-0.1588f).epsilon(1e-2));
    REQUIRE(it[2] == Catch::Approx(0.0f));
    REQUIRE(it[3] == Catch::Approx(0.8412f).epsilon(1e-2));
}

TEST_CASE("cpu::log_softmax on non-contiguous (transposed) input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_x = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    // Transposed {3,2}, strides {1,3}: flat logical order = storage[0],storage[3],storage[1],storage[4],storage[2],storage[5]
    // Row 0: flat 0,1 → storage[0],storage[3]
    // Row 1: flat 2,3 → storage[1],storage[4]
    // Row 2: flat 4,5 → storage[2],storage[5]
    // Want rows: [1,2], [3,4], [5,6] — same as contiguous {3,2} with [1,2,3,4,5,6]
    // So storage = [1, 3, 5, 2, 4, 6]
    float x_vals[] = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
    std::memcpy(storage_x->data, x_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor x(tt, storage_x, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::log_softmax(out, x);
    REQUIRE(result);

    // Contiguous reference with same row data
    auto x_ref = Tensor::empty(rt, {3, 2});
    float ref_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::memcpy(x_ref.data<float>(), ref_vals, 6 * sizeof(float));
    auto out_ref = Tensor::empty(rt, {3, 2});
    cpu::log_softmax(out_ref, x_ref);

    TensorIterator<float> it(out);
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(it[i] == Catch::Approx(out_ref.data<float>()[i]).epsilon(1e-5f));
    }
}

TEST_CASE("cpu::softmax on non-contiguous (transposed) input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_x = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    // Same layout as log_softmax test
    float x_vals[] = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
    std::memcpy(storage_x->data, x_vals, 6 * sizeof(float));
    std::memset(storage_out->data, 0, 6 * sizeof(float));

    TensorType tt({3, 2}, {1, 3}, DType::Float32);
    Tensor x(tt, storage_x, false, 0);
    Tensor out(tt, storage_out, false, 0);

    auto result = cpu::softmax(out, x);
    REQUIRE(result);

    auto x_ref = Tensor::empty(rt, {3, 2});
    float ref_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::memcpy(x_ref.data<float>(), ref_vals, 6 * sizeof(float));
    auto out_ref = Tensor::empty(rt, {3, 2});
    cpu::softmax(out_ref, x_ref);

    TensorIterator<float> it(out);
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(it[i] == Catch::Approx(out_ref.data<float>()[i]).epsilon(1e-5f));
    }
}

// ── Non-contiguous reduction kernel tests ─────────────────────────────

TEST_CASE("cpu::matmul on non-contiguous (transposed) A", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_a = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float a_vals[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    std::memcpy(storage_a->data, a_vals, 6 * sizeof(float));
    TensorType nc_type({2, 3}, {1, 2}, DType::Float32);
    Tensor a_nc(nc_type, storage_a, false, 0);

    auto b = rt.empty({3, 2});
    float b_vals[] = {7,8,9,10,11,12};
    std::memcpy(b.data<float>(), b_vals, 6 * sizeof(float));

    auto a_ref = rt.empty({2, 3});
    float a_ref_vals[] = {1,2,3,4,5,6};
    std::memcpy(a_ref.data<float>(), a_ref_vals, 6 * sizeof(float));
    auto out_ref = rt.empty({2, 2});
    cpu::matmul(out_ref, a_ref, b);

    auto storage_out = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    Tensor out_nc(TensorType::contiguous({2, 2}, DType::Float32), storage_out, false, 0);
    auto result = cpu::matmul(out_nc, a_nc, b);
    REQUIRE(result);

    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE(out_nc.data<float>()[i] == Catch::Approx(out_ref.data<float>()[i]));
    }
}

TEST_CASE("cpu::conv2d on non-contiguous input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_inp = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    float inp_vals[] = {1.0f, 3.0f, 2.0f, 4.0f};
    std::memcpy(storage_inp->data, inp_vals, 4 * sizeof(float));
    TensorType inp_nc_type({1, 1, 2, 2}, {4, 4, 1, 2}, DType::Float32);
    Tensor inp_nc(inp_nc_type, storage_inp, false, 0);

    auto inp_ref = rt.empty({1, 1, 2, 2});
    float inp_ref_vals[] = {1,2,3,4};
    std::memcpy(inp_ref.data<float>(), inp_ref_vals, 4 * sizeof(float));

    auto weight = rt.ones({1, 1, 1, 1});
    weight.data<float>()[0] = 2.0f;

    auto out_ref = rt.empty({1, 1, 2, 2});
    cpu::conv2d(out_ref, inp_ref, weight, 1, 0);

    auto storage_out = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    Tensor out_nc(TensorType::contiguous({1, 1, 2, 2}, DType::Float32), storage_out, false, 0);
    auto result = cpu::conv2d(out_nc, inp_nc, weight, 1, 0);
    REQUIRE(result);

    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE(out_nc.data<float>()[i] == Catch::Approx(out_ref.data<float>()[i]));
    }
}

TEST_CASE("cpu::reduce_mean on non-contiguous input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_inp = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float inp_vals[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    std::memcpy(storage_inp->data, inp_vals, 6 * sizeof(float));
    TensorType nc_type({2, 3}, {1, 2}, DType::Float32);
    Tensor inp_nc(nc_type, storage_inp, false, 0);

    auto inp_ref = rt.empty({2, 3});
    float ref_vals[] = {1,2,3,4,5,6};
    std::memcpy(inp_ref.data<float>(), ref_vals, 6 * sizeof(float));

    auto out_ref = rt.empty({2});
    cpu::reduce_mean(out_ref, inp_ref, {1});

    auto storage_out = rt.allocator().allocate(TensorType::contiguous({2}, DType::Float32));
    Tensor out_nc(TensorType::contiguous({2}, DType::Float32), storage_out, false, 0);
    auto result = cpu::reduce_mean(out_nc, inp_nc, {1});
    REQUIRE(result);

    REQUIRE(out_nc.data<float>()[0] == Catch::Approx(out_ref.data<float>()[0]));
    REQUIRE(out_nc.data<float>()[1] == Catch::Approx(out_ref.data<float>()[1]));
}

TEST_CASE("cpu::batchnorm on non-contiguous input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_inp = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    float inp_vals[] = {1.0f, 3.0f, 2.0f, 4.0f};
    std::memcpy(storage_inp->data, inp_vals, 4 * sizeof(float));
    TensorType inp_nc_type({1, 2, 1, 2}, {4, 1, 2, 2}, DType::Float32);
    Tensor inp_nc(inp_nc_type, storage_inp, false, 0);

    auto gamma = rt.ones({2});
    auto beta = rt.zeros({2});
    auto running_mean = rt.zeros({2});
    auto running_var = rt.ones({2});

    auto inp_ref = rt.empty({1, 2, 1, 2});
    float ref_vals[] = {1,2,3,4};
    std::memcpy(inp_ref.data<float>(), ref_vals, 4 * sizeof(float));
    auto out_ref = rt.empty({1, 2, 1, 2});
    cpu::batchnorm(out_ref, inp_ref, gamma, beta, running_mean, running_var, 0.9f, 1e-5f, false);

    auto storage_out = rt.allocator().allocate(TensorType::contiguous({4}, DType::Float32));
    Tensor out_nc(TensorType::contiguous({1, 2, 1, 2}, DType::Float32), storage_out, false, 0);
    auto result = cpu::batchnorm(out_nc, inp_nc, gamma, beta, running_mean, running_var, 0.9f, 1e-5f, false);
    REQUIRE(result);

    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE(out_nc.data<float>()[i] == Catch::Approx(out_ref.data<float>()[i]));
    }
}

TEST_CASE("cpu::layernorm on non-contiguous input", "[backend][cpu][noncontig]") {
    Runtime rt;
    auto storage_inp = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    float inp_vals[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    std::memcpy(storage_inp->data, inp_vals, 6 * sizeof(float));
    TensorType nc_type({2, 3}, {1, 2}, DType::Float32);
    Tensor inp_nc(nc_type, storage_inp, false, 0);

    auto gamma = rt.ones({3});
    auto beta = rt.zeros({3});

    auto inp_ref = rt.empty({2, 3});
    float ref_vals[] = {1,2,3,4,5,6};
    std::memcpy(inp_ref.data<float>(), ref_vals, 6 * sizeof(float));
    auto out_ref = rt.empty({2, 3});
    cpu::layernorm(out_ref, inp_ref, gamma, beta, 1e-5f);

    auto storage_out = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    Tensor out_nc(TensorType::contiguous({2, 3}, DType::Float32), storage_out, false, 0);
    auto result = cpu::layernorm(out_nc, inp_nc, gamma, beta, 1e-5f);
    REQUIRE(result);

    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(out_nc.data<float>()[i] == Catch::Approx(out_ref.data<float>()[i]).epsilon(1e-4f));
    }
}

// ── mul_scalar ──────────────────────────────────────────────────────────

TEST_CASE("cpu::mul_scalar multiplies each element by scalar", "[backend][cpu]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    auto out = Tensor::empty(rt, {2, 3});

    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));

    auto result = cpu::mul_scalar(out, x, 3.0f);
    REQUIRE(result);

    float expected[] = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(out.data<float>()[i] == Catch::Approx(expected[i]));
    }
}

TEST_CASE("cpu::mul_scalar rejects shape mismatch", "[backend][cpu]") {
    Runtime rt;
    auto x = rt.ones({2, 3});
    auto out = rt.empty({3, 2});

    auto result = cpu::mul_scalar(out, x, 2.0f);
    REQUIRE_FALSE(result);
}

// ── div_scalar ──────────────────────────────────────────────────────────

TEST_CASE("cpu::div_scalar divides each element by scalar", "[backend][cpu]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    auto out = Tensor::empty(rt, {2, 3});

    float x_data[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));

    auto result = cpu::div_scalar(out, x, 2.0f);
    REQUIRE(result);

    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(out.data<float>()[i] == Catch::Approx(expected[i]));
    }
}

TEST_CASE("cpu::div_scalar rejects division by zero", "[backend][cpu]") {
    Runtime rt;
    auto x = rt.ones({2, 3});
    auto out = rt.empty({2, 3});

    auto result = cpu::div_scalar(out, x, 0.0f);
    REQUIRE_FALSE(result);
}

TEST_CASE("cpu::div_scalar rejects shape mismatch", "[backend][cpu]") {
    Runtime rt;
    auto x = rt.ones({2, 3});
    auto out = rt.empty({3, 2});

    auto result = cpu::div_scalar(out, x, 2.0f);
    REQUIRE_FALSE(result);
}
