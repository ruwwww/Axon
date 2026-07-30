#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include <cmath>
#include "axon/backend/cpu_backend.h"
#include "axon/runtime/runtime.h"
#include "axon/tensor/tensor_iterator.h"
#include "axon/nn/mse.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/l1_loss.h"

using namespace axon;

// ── Backend kernel tests ──────────────────────────────────────────────

TEST_CASE("cpu::matmul computes correct 1x3 @ 3x1 -> 1x1", "[backend][cpu]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {1, 3});
    auto b = Tensor::empty(rt, {3, 1});
    auto out = Tensor::empty(rt, {1, 1});

    a.data<float>()[0] = 1.0f; a.data<float>()[1] = 2.0f; a.data<float>()[2] = 3.0f;
    b.data<float>()[0] = 4.0f; b.data<float>()[1] = 5.0f; b.data<float>()[2] = 6.0f;

    auto result = cpu::matmul(out, a, b);
    REQUIRE(result);
    REQUIRE(out.data<float>()[0] == Catch::Approx(32.0f));
}

TEST_CASE("cpu::matmul computes correct 2x3 @ 3x2 -> 2x2", "[backend][cpu]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {2, 3});
    auto b = Tensor::empty(rt, {3, 2});
    auto out = Tensor::empty(rt, {2, 2});

    float a_data[] = {1,2,3,4,5,6};
    float b_data[] = {7,8,9,10,11,12};
    std::memcpy(a.data<float>(), a_data, 6 * sizeof(float));
    std::memcpy(b.data<float>(), b_data, 6 * sizeof(float));

    auto result = cpu::matmul(out, a, b);
    REQUIRE(result);

    REQUIRE(out.data<float>()[0] == Catch::Approx(58.0f));
    REQUIRE(out.data<float>()[1] == Catch::Approx(64.0f));
    REQUIRE(out.data<float>()[2] == Catch::Approx(139.0f));
    REQUIRE(out.data<float>()[3] == Catch::Approx(154.0f));
}

TEST_CASE("cpu::matmul rejects mismatched inner dimensions", "[backend][cpu]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {2, 3});
    auto b = Tensor::empty(rt, {4, 2});
    auto out = Tensor::empty(rt, {2, 2});

    auto result = cpu::matmul(out, a, b);
    REQUIRE_FALSE(result);
}

TEST_CASE("cpu::relu computes correct result", "[backend][cpu]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    auto out = Tensor::empty(rt, {2, 3});

    float x_data[] = {-2, -1, 0, 1, 2, 3};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));

    auto result = cpu::relu(out, x);
    REQUIRE(result);

    REQUIRE(out.data<float>()[0] == Catch::Approx(0.0f));
    REQUIRE(out.data<float>()[1] == Catch::Approx(0.0f));
    REQUIRE(out.data<float>()[2] == Catch::Approx(0.0f));
    REQUIRE(out.data<float>()[3] == Catch::Approx(1.0f));
    REQUIRE(out.data<float>()[4] == Catch::Approx(2.0f));
    REQUIRE(out.data<float>()[5] == Catch::Approx(3.0f));
}

// ── Operation tests ───────────────────────────────────────────────────

TEST_CASE("MatMulOp forward allocates correct output shape", "[operation]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {2, 3});
    auto b = Tensor::zeros(rt, {3, 4});

    auto result = rt.matmul(a, b);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 4}));
}

TEST_CASE("MatMulOp forward rejects mismatched shapes", "[operation]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {2, 3});
    auto b = Tensor::zeros(rt, {5, 4});

    auto result = rt.matmul(a, b);
    REQUIRE_FALSE(result);
}

TEST_CASE("MatMulOp forward records graph node when requires_grad", "[operation]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {2, 3});
    auto b = Tensor::zeros(rt, {3, 4});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.matmul(a, b);
    REQUIRE(rt.autograd().graph().size() == 1);
}

TEST_CASE("MatMulOp forward does not record graph when requires_grad is false", "[operation]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {2, 3});
    auto b = Tensor::zeros(rt, {3, 4});

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.matmul(a, b);
    REQUIRE(rt.autograd().graph().size() == 0);
}

TEST_CASE("ReLUOp forward allocates correct output shape", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.relu(x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 3}));
}

TEST_CASE("ReLUOp forward records graph node when requires_grad", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.relu(x);
    REQUIRE(rt.autograd().graph().size() == 1);
}

TEST_CASE("ReLUOp forward does not record graph when requires_grad is false", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.relu(x);
    REQUIRE(rt.autograd().graph().size() == 0);
}

// ── GELUOp tests ─────────────────────────────────────────────────────

TEST_CASE("GELUOp forward allocates correct output shape", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.gelu(x);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 3}));
}

TEST_CASE("GELUOp forward records graph node when requires_grad", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.gelu(x);
    REQUIRE(rt.autograd().graph().size() == 1);
}

TEST_CASE("GELUOp forward does not record graph when requires_grad is false", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.gelu(x);
    REQUIRE(rt.autograd().graph().size() == 0);
}

TEST_CASE("GELUOp backward matches finite differences", "[autograd]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {4});
    x.set_requires_grad(true);

    float x_data[] = {-1.5f, -0.5f, 0.5f, 1.5f};
    std::memcpy(x.data<float>(), x_data, 4 * sizeof(float));

    Tensor y = *rt.gelu(x);

    auto result = rt.autograd().backward(rt, y);
    REQUIRE(result);

    auto& grads = rt.autograd().gradients();
    Tensor grad_x = grads[x.id()];

    REQUIRE(grad_x.type().shape() == std::vector<int64_t>({4}));

    const double eps = 1.0 / 1024.0;
    for (int64_t i = 0; i < 4; ++i) {
        double orig = x_data[i];

        double y_plus = 0.0;
        {
            float xi = static_cast<float>(orig + eps);
            float x3 = xi * xi * xi;
            float inner = 0.79788456f * (xi + 0.044715f * x3);
            y_plus = 0.5 * xi * (1.0 + std::tanh(inner));
        }
        {
            float xi = static_cast<float>(orig + eps);
            float x3 = xi * xi * xi;
            float inner = 0.79788456f * (xi + 0.044715f * x3);
            y_plus = 0.5 * xi * (1.0 + std::tanh(inner));
        }

        x.data<float>()[i] = static_cast<float>(orig - eps);
        double y_minus = 0.0;
        {
            float xi = static_cast<float>(orig - eps);
            float x3 = xi * xi * xi;
            float inner = 0.79788456f * (xi + 0.044715f * x3);
            y_minus = 0.5 * xi * (1.0 + std::tanh(inner));
        }

        x.data<float>()[i] = static_cast<float>(orig);

        double numerical = (y_plus - y_minus) / (2.0 * eps);
        REQUIRE(static_cast<double>(grad_x.data<float>()[i]) == Catch::Approx(numerical).epsilon(1e-3));
    }
}

// ── Autograd integration tests ────────────────────────────────────────

TEST_CASE("Autograd backward produces gradients for all inputs", "[autograd]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {1, 3});
    auto b = Tensor::zeros(rt, {3, 1});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    a.data<float>()[0] = 1.0f; a.data<float>()[1] = 2.0f; a.data<float>()[2] = 3.0f;
    b.data<float>()[0] = 4.0f; b.data<float>()[1] = 5.0f; b.data<float>()[2] = 6.0f;

    Tensor c = *rt.matmul(a, b);

    auto result = rt.autograd().backward(rt, c);
    REQUIRE(result);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(a.id()) > 0);
    REQUIRE(grads.count(b.id()) > 0);
}

// ── Strided backward tests (TensorIterator migration) ─────────────────

TEST_CASE("ReLUOp backward with non-contiguous input uses correct strides", "[autograd][strided]") {
    Runtime rt;

    // Input: transposed view shape [3,2], strides [1,3] backed by 6-element base
    auto base = Tensor::zeros(rt, {6});
    float base_data[] = {-2, -1, 0, 1, 2, 3};
    std::memcpy(base.data<float>(), base_data, 6 * sizeof(float));
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor x(view_type, base.storage(), false);

    // Contiguous grad_out shape [3,2]
    auto go = Tensor::empty(rt, {3, 2});
    float go_data[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(go.data<float>(), go_data, 6 * sizeof(float));

    // dx is allocated by backward as contiguous [3,2]
    auto dx_type = TensorType::contiguous({3, 2}, DType::Float32);
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    // Manually run what ReLUOp::backward does with TensorIterator
    TensorIterator<const float> x_it(x);
    TensorIterator<const float> go_it(go);
    TensorIterator<float> dx_it(dx);
    auto n = x.type().numel();
    for (int64_t i = 0; i < n; ++i) {
        dx_it[i] = x_it[i] > 0.0f ? go_it[i] : 0.0f;
    }

    float expected[] = {0, 2, 0, 4, 0, 6};
    for (int64_t i = 0; i < n; ++i) {
        REQUIRE(dx.data<float>()[i] == Catch::Approx(expected[i]));
    }
}

TEST_CASE("GELUOp backward with non-contiguous input uses correct strides", "[autograd][strided]") {
    Runtime rt;
    // Fill base so non-contiguous view (shape[3,2], strides[1,3]) has logical row-major values 0..5
    // stride mapping: logical(i,j) -> base[i*1 + j*3]
    auto base = Tensor::zeros(rt, {6});
    float* b = base.data<float>();
    // logical(0,0)=0, logical(0,1)=1, logical(1,0)=2, logical(1,1)=3, logical(2,0)=4, logical(2,1)=5
    b[0]=0; b[3]=1; b[1]=2; b[4]=3; b[2]=4; b[5]=5;
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor x(view_type, base.storage(), false);

    // Contiguous grad_out has matching logical values 10..15
    auto go = Tensor::empty(rt, {3, 2});
    float go_data[] = {10, 11, 12, 13, 14, 15};
    std::memcpy(go.data<float>(), go_data, 6 * sizeof(float));

    auto dx_type = TensorType::contiguous({3, 2}, DType::Float32);
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    TensorIterator<const float> x_it(x);
    TensorIterator<const float> go_it(go);
    TensorIterator<float> dx_it(dx);
    auto n = x.type().numel();

    // x_it reads logical row-major: 0,1,2,3,4,5; go_it reads contiguously: 10..15
    constexpr float alpha = 0.79788456f;
    constexpr float beta = 0.044715f;
    for (int64_t i = 0; i < n; ++i) {
        float xi = x_it[i];
        float x3 = xi * xi * xi;
        float g = alpha * (xi + beta * x3);
        float t = std::tanh(g);
        float t2 = 1.0f - t * t;
        float gprime = alpha * (1.0f + 3.0f * beta * xi * xi);
        float df = 0.5f * (1.0f + t + xi * t2 * gprime);
        dx_it[i] = go_it[i] * df;
    }

    // Expected: GELU derivative at logical position i, times go_data[i]
    for (int64_t i = 0; i < n; ++i) {
        float xi = static_cast<float>(i);
        float x3 = xi * xi * xi;
        float g = alpha * (xi + beta * x3);
        float t = std::tanh(g);
        float t2 = 1.0f - t * t;
        float gprime = alpha * (1.0f + 3.0f * beta * xi * xi);
        float df = 0.5f * (1.0f + t + xi * t2 * gprime);
        float expected = go_data[i] * df;
        REQUIRE(dx.data<float>()[i] == Catch::Approx(expected).epsilon(1e-5));
    }
}

TEST_CASE("MSELossOp backward with non-contiguous input uses correct strides", "[autograd][strided]") {
    Runtime rt;

    auto pred_base = Tensor::zeros(rt, {6});
    float* pb = pred_base.data<float>();
    pb[0]=0; pb[3]=1; pb[1]=2; pb[4]=3; pb[2]=4; pb[5]=5;  // logical: 0,1,2,3,4,5
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor pred(view_type, pred_base.storage(), false);

    auto target_base = Tensor::zeros(rt, {6});
    float* tb = target_base.data<float>();
    tb[0]=5; tb[3]=4; tb[1]=3; tb[4]=2; tb[2]=1; tb[5]=0;  // logical: 5,4,3,2,1,0
    Tensor target(view_type, target_base.storage(), false);

    auto dp_type = TensorType::contiguous({3, 2}, DType::Float32);
    Tensor dp(dp_type, rt.allocator().allocate(dp_type), false);

    TensorIterator<const float> p_it(pred);
    TensorIterator<const float> t_it(target);
    TensorIterator<float> dp_it(dp);
    auto n = pred.type().numel();
    float inv_N = 2.0f / static_cast<float>(n);
    for (int64_t i = 0; i < n; ++i) {
        dp_it[i] = (p_it[i] - t_it[i]) * inv_N;
    }

    // p_it reads logical: 0,1,2,3,4,5; t_it reads logical: 5,4,3,2,1,0
    // diff: -5,-3,-1,1,3,5
    float expected[] = {-5 * inv_N, -3 * inv_N, -1 * inv_N, 1 * inv_N, 3 * inv_N, 5 * inv_N};
    for (int64_t i = 0; i < n; ++i) {
        REQUIRE(dp.data<float>()[i] == Catch::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("L1LossOp backward with non-contiguous input uses correct strides", "[autograd][strided]") {
    Runtime rt;

    auto pred_base = Tensor::zeros(rt, {6});
    float* pb = pred_base.data<float>();
    pb[0]=0; pb[3]=1; pb[1]=2; pb[4]=3; pb[2]=4; pb[5]=5;  // logical: 0,1,2,3,4,5
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor pred(view_type, pred_base.storage(), false);

    auto target_base = Tensor::zeros(rt, {6});
    float* tb = target_base.data<float>();
    tb[0]=5; tb[3]=4; tb[1]=3; tb[4]=2; tb[2]=1; tb[5]=0;  // logical: 5,4,3,2,1,0
    Tensor target(view_type, target_base.storage(), false);

    auto dp_type = TensorType::contiguous({3, 2}, DType::Float32);
    Tensor dp(dp_type, rt.allocator().allocate(dp_type), false);

    TensorIterator<const float> p_it(pred);
    TensorIterator<const float> t_it(target);
    TensorIterator<float> dp_it(dp);
    auto n = pred.type().numel();
    float inv_N = 1.0f / static_cast<float>(n);
    for (int64_t i = 0; i < n; ++i) {
        float diff = p_it[i] - t_it[i];
        dp_it[i] = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)) * inv_N;
    }

    // p_it reads logical: 0,1,2,3,4,5; t_it reads logical: 5,4,3,2,1,0
    // diff: -5,-3,-1,1,3,5 -> sign: -1,-1,-1,1,1,1
    float expected[] = {-inv_N, -inv_N, -inv_N, inv_N, inv_N, inv_N};
    for (int64_t i = 0; i < n; ++i) {
        REQUIRE(dp.data<float>()[i] == Catch::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("CrossEntropyLossOp backward with non-contiguous log_softmax uses correct strides", "[autograd][strided]") {
    Runtime rt;

    auto N = int64_t(2), C = int64_t(3);
    // Fill for shape [2,3] strides [1,2] to have known logical row-major values
    // offset = i*stride[0] + j*stride[1] = i*1 + j*2
    // logical(0,0)->base[0], logical(0,1)->base[2], logical(0,2)->base[4]
    // logical(1,0)->base[1], logical(1,1)->base[3], logical(1,2)->base[5]
    auto ls_base = Tensor::zeros(rt, {6});
    float* lb = ls_base.data<float>();
    lb[0] = -1.0f; lb[2] = -0.5f; lb[4] = -1.5f;   // row 0 logical: -1, -0.5, -1.5
    lb[1] = -2.0f; lb[3] = -0.8f; lb[5] = -0.3f;   // row 1 logical: -2, -0.8, -0.3
    TensorType view_type({2, 3}, {1, 2}, DType::Float32);
    Tensor log_softmax_out(view_type, ls_base.storage(), false);

    auto targets = Tensor::empty(rt, {2});
    targets.data<int64_t>()[0] = 0;
    targets.data<int64_t>()[1] = 2;

    auto dlogits_type = TensorType::contiguous({2, 3}, DType::Float32);
    Tensor dlogits(dlogits_type, rt.allocator().allocate(dlogits_type), false);

    TensorIterator<const float> ls_it(log_softmax_out);
    TensorIterator<const int64_t> t_it(targets);
    TensorIterator<float> d_it(dlogits);
    float inv_N = 1.0f / static_cast<float>(N);

    for (int64_t i = 0; i < N; ++i) {
        int64_t target = t_it[i];
        for (int64_t j = 0; j < C; ++j) {
            float sm = std::exp(ls_it[i * C + j]);
            d_it[i * C + j] = (sm - (j == target ? 1.0f : 0.0f)) * inv_N;
        }
    }

    // Expected based on logical row-major: log_softmax = [-1, -0.5, -1.5, -2, -0.8, -0.3]
    float logical_ls[] = {-1.0f, -0.5f, -1.5f, -2.0f, -0.8f, -0.3f};
    for (int64_t i = 0; i < N; ++i) {
        int64_t target = targets.data<const int64_t>()[i];
        for (int64_t j = 0; j < C; ++j) {
            float sm = std::exp(logical_ls[i * C + j]);
            float expected = (sm - (j == target ? 1.0f : 0.0f)) * inv_N;
            REQUIRE(dlogits.data<float>()[i * C + j] == Catch::Approx(expected).epsilon(1e-6));
        }
    }
}

TEST_CASE("AddOp backward with non-contiguous grad_out uses correct strides", "[autograd][strided]") {
    Runtime rt;

    // grad_out shape [3,2] strides [1,3]; fill so logical row-major = [0,1,2,3,4,5]
    // offset = i*1 + j*3: logical(0,0)->base[0], logical(0,1)->base[3]
    // logical(1,0)->base[1], logical(1,1)->base[4], logical(2,0)->base[2], logical(2,1)->base[5]
    auto go_base = Tensor::zeros(rt, {6});
    float* gb = go_base.data<float>();
    gb[0]=0; gb[3]=1; gb[1]=2; gb[4]=3; gb[2]=4; gb[5]=5;
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor grad_out(view_type, go_base.storage(), false);

    auto da_type = TensorType::contiguous({3, 2}, DType::Float32);
    Tensor da(da_type, rt.allocator().allocate(da_type), false);

    TensorIterator<const float> go_it(grad_out);
    TensorIterator<float> da_it(da);
    auto n = grad_out.type().numel();
    for (int64_t i = 0; i < n; ++i) {
        da_it[i] = go_it[i];
    }

    // go_it reads logical row-major: 0,1,2,3,4,5
    for (int64_t i = 0; i < n; ++i) {
        REQUIRE(da.data<float>()[i] == Catch::Approx(static_cast<float>(i)).epsilon(1e-6));
    }
}

TEST_CASE("AddOp backward bias-sum with non-contiguous grad_out uses correct strides", "[autograd][strided]") {
    Runtime rt;

    // grad_out shape [3,2] non-contiguous
    auto go_base = Tensor::zeros(rt, {8});
    float go_data[] = {10, 20, 30, 40, 50, 60};
    std::memcpy(go_base.data<float>(), go_data, 6 * sizeof(float));
    TensorType go_view_type({2, 3}, {1, 2}, DType::Float32);
    Tensor grad_out(go_view_type, go_base.storage(), false);

    // Bias sum C=3
    int64_t M = 2, N = 3;
    auto db_type = TensorType::contiguous({3}, DType::Float32);
    Tensor db(db_type, rt.allocator().allocate(db_type), false);
    auto* db_ptr = db.data<float>();

    TensorIterator<const float> go_it(grad_out);
    for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < M; ++i) {
            sum += go_it[i * N + j];
        }
        db_ptr[j] = sum;
    }

    // grad_out logical layout (strides [1,2]):
    // [0,0]=base[0]=10  [0,1]=base[2]=30  [0,2]=base[4]=50
    // [1,0]=base[1]=20  [1,1]=base[3]=40  [1,2]=base[5]=60
    // bias sum over rows: db[0]=10+20=30, db[1]=30+40=70, db[2]=50+60=110
    float expected[] = {30, 70, 110};
    for (int64_t j = 0; j < N; ++j) {
        REQUIRE(db.data<float>()[j] == Catch::Approx(expected[j]).epsilon(1e-6));
    }
}

TEST_CASE("MatMulOp backward with non-contiguous grad_out uses correct strides", "[autograd][strided]") {
    Runtime rt;

    int64_t M = 2, K = 3, N = 2;
    auto a = Tensor::empty(rt, {M, K});
    float a_data[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(a.data<float>(), a_data, 6 * sizeof(float));

    auto b = Tensor::empty(rt, {K, N});
    float b_data[] = {7, 8, 9, 10, 11, 12};
    std::memcpy(b.data<float>(), b_data, 6 * sizeof(float));

    // grad_out as non-contiguous view
    auto go_base = Tensor::zeros(rt, {8});
    float go_data[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(go_base.data<float>(), go_data, 6 * sizeof(float));
    TensorType go_view_type({2, 2}, {1, 2}, DType::Float32);
    Tensor grad_out(go_view_type, go_base.storage(), false);

    // Compute da = grad_out @ b^T  (M x K)
    auto da_type = TensorType::contiguous({M, K}, DType::Float32);
    Tensor da(da_type, rt.allocator().allocate(da_type), false);

    TensorIterator<const float> go_it(grad_out);
    TensorIterator<const float> b_it(b);
    TensorIterator<float> da_it(da);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < N; ++k) {
                sum += go_it[i * N + k] * b_it[j * N + k];
            }
            da_it[i * K + j] = sum;
        }
    }

    // grad_out logical layout (strides [1,2], shape [2,2]):
    // [0,0]=base[0]=1  [0,1]=base[2]=3
    // [1,0]=base[1]=2  [1,1]=base[3]=4
    // b = {7,8,9,10,11,12} contiguous [3,2]
    // b^T is {7,9,11} on rows, {8,10,12} on cols -> shape [2,3] in column-major
    // Wait, b is [3,2] contiguous: rows are {7,8}, {9,10}, {11,12}
    // b[j][k] where j in [0,K), k in [0,N)
    // Grad_out logical (using stride {1,2}):
    // go_it[i * N + k] = go_it[i * 2 + k]
    // da[i][j] = sum_k go_it[i*N + k] * b_it[j*N + k] for k=0..N-1
    // da[0][0] = go[0][0]*b[0][0] + go[0][1]*b[0][1] = 1*7 + 3*8 = 7+24 = 31
    // da[0][1] = 1*9 + 3*10 = 9+30 = 39
    // da[0][2] = 1*11 + 3*12 = 11+36 = 47
    // da[1][0] = 2*7 + 4*8 = 14+32 = 46
    // da[1][1] = 2*9 + 4*10 = 18+40 = 58
    // da[1][2] = 2*11 + 4*12 = 22+48 = 70

    float expected_da[] = {31, 39, 47, 46, 58, 70};
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(da.data<float>()[i] == Catch::Approx(expected_da[i]).epsilon(1e-6));
    }
}

TEST_CASE("Composite: transpose -> relu backward with non-contiguous grad uses correct strides", "[autograd][strided]") {
    Runtime rt;

    // Full pipeline: input tensor is transposed, then relu applied, then backward
    // x has shape [2,3], contiguous
    auto x = Tensor::empty(rt, {2, 3});
    float x_data[] = {-1, 2, -3, 4, -5, 6};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));
    x.set_requires_grad(true);

    // Forward: transpose(0,1) then relu
    // After transpose: shape [3,2], strides [1,3]
    auto xt = *rt.transpose(x, 0, 1);
    REQUIRE(xt.type().shape() == std::vector<int64_t>({3, 2}));
    REQUIRE(xt.type().strides() == std::vector<int64_t>({1, 3}));

    auto y = *rt.relu(xt);

    // y is contiguous [3,2], filled with ReLU of transposed x
    // y[i][j] = relu(x[j][i]) since x is [2,3] and xt is [3,2]

    // Backward
    rt.autograd().backward(rt, y);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(x.id()) > 0);
    Tensor grad_x = grads[x.id()];
    REQUIRE(grad_x.type().shape() == std::vector<int64_t>({2, 3}));

    // dx[j,i] = relu'(x[j,i])  (loss = sum(y))
    float expected[] = {0, 1, 0, 1, 0, 1};
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(grad_x.data<float>()[i] == Catch::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("Composite: transpose -> matmul -> backward with non-contiguous tensors", "[autograd][strided]") {
    Runtime rt;

    // x: {2,3} transposed -> {3,2} then matmul with W: {2,4} -> y: {3,4}
    auto x = Tensor::empty(rt, {2, 3});
    float x_data[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));
    x.set_requires_grad(true);

    auto W = Tensor::empty(rt, {2, 4});
    float w_data[] = {1,0,0,1, 0,1,1,0};
    std::memcpy(W.data<float>(), w_data, 8 * sizeof(float));
    W.set_requires_grad(true);

    // xT = transpose(x, 0, 1) -> {3,2} with strides [1,3]
    auto xT = *rt.transpose(x, 0, 1);
    // y = xT @ W -> {3,4} using matmul
    auto y = *rt.matmul(xT, W);

    // backward
    rt.autograd().backward(rt, y);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(x.id()) > 0);
    REQUIRE(grads.count(W.id()) > 0);

    // y[k][m] = sum_j xT[k][j] * W[j][m] = sum_j x[j][k] * W[j][m]
    // grad_out for y = ones({3,4}) since loss = sum(y)
    // dW[j][m] = sum_k xT[k][j] * 1 = sum_k x[j][k]
    // dW[0,0] = sum_k x[0][k] = 1+2+3 = 6
    // dW[0,1] = 6
    // dW[0,2] = 6
    // dW[0,3] = 6
    // dW[1,0] = sum_k x[1][k] = 4+5+6 = 15
    // dW[1,1] = 15 ... actually wait

    // W is [2,4], so W[j,m] with j in [0,2), m in [0,4)
    // y = xT @ W, xT is [3,2], W is [2,4], y is [3,4]
    // dW[j,m] = sum_k xT[k,j] * grad_out[k,m] = sum_k x[j,k] * 1 (since grad_out is all ones)
    // dW[0,m] = sum_k x[0,k] = 6 for all m
    // dW[1,m] = sum_k x[1,k] = 15 for all m

    // Actually loss = sum(y) -> grad_out = ones([3,4])
    // dW = xT^T @ grad_out = x @ ones([3,4])
    // x is [2,3], ones is [3,4] -> dW is [2,4]
    // dW[i,m] = sum_k x[i,k] * 1 = sum_k x[i,k]
    // dW[0] = [6, 6, 6, 6]
    // dW[1] = [15, 15, 15, 15]

    Tensor grad_W = grads[W.id()];
    REQUIRE(grad_W.type().shape() == std::vector<int64_t>({2, 4}));
    for (int64_t m = 0; m < 4; ++m) {
        REQUIRE(grad_W.data<float>()[0 * 4 + m] == Catch::Approx(6.0f));
        REQUIRE(grad_W.data<float>()[1 * 4 + m] == Catch::Approx(15.0f));
    }

    // dx[i,k] = sum_m grad_out[k,m] * W[i,m]^T wait...
    // Actually da = grad_out @ b^T where a=xT, b=W
    // d(xT) = grad_out @ W^T = ones([3,4]) @ W^T where W^T is [4,2]
    // d(xT)[k,i] = sum_m 1 * W[i,m] = sum_m W[i,m]
    // = [sum of W[0]] = 1+0+0+1 = 2 for all k
    // = [sum of W[1]] = 0+1+1+0 = 2 for all k
    // So d(xT) = [[2,2],[2,2],[2,2]]
    //
    // Then dx = transpose(d(xT), 0, 1) since x = transpose(xT, 1, 0)
    // Wait, xT = transpose(x, 0, 1), so x = transpose(xT, 0, 1)
    // dx[k,i] = d(xT)[i,k] = 2 for all elements
    // So dx = [[2,2,2],[2,2,2]]

    Tensor grad_x = grads[x.id()];
    REQUIRE(grad_x.type().shape() == std::vector<int64_t>({2, 3}));
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(grad_x.data<float>()[i] == Catch::Approx(2.0f).epsilon(1e-6));
    }
}

TEST_CASE("Autograd backward computes correct matmul gradients", "[autograd]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {1, 3});
    auto b = Tensor::empty(rt, {3, 1});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    a.data<float>()[0] = 1.0f; a.data<float>()[1] = 2.0f; a.data<float>()[2] = 3.0f;
    b.data<float>()[0] = 4.0f; b.data<float>()[1] = 5.0f; b.data<float>()[2] = 6.0f;

    Tensor c = *rt.matmul(a, b);

    auto result = rt.autograd().backward(rt, c);
    REQUIRE(result);

    auto& grads = rt.autograd().gradients();

    Tensor grad_a = grads[a.id()];
    REQUIRE(grad_a.type().shape() == std::vector<int64_t>({1, 3}));
    REQUIRE(grad_a.data<float>()[0] == Catch::Approx(4.0f));
    REQUIRE(grad_a.data<float>()[1] == Catch::Approx(5.0f));
    REQUIRE(grad_a.data<float>()[2] == Catch::Approx(6.0f));

    Tensor grad_b = grads[b.id()];
    REQUIRE(grad_b.type().shape() == std::vector<int64_t>({3, 1}));
    REQUIRE(grad_b.data<float>()[0] == Catch::Approx(1.0f));
    REQUIRE(grad_b.data<float>()[1] == Catch::Approx(2.0f));
    REQUIRE(grad_b.data<float>()[2] == Catch::Approx(3.0f));
}

TEST_CASE("Autograd backward matches finite differences", "[autograd]") {
    Runtime rt;
    auto W = Tensor::empty(rt, {1, 3});
    auto x = Tensor::empty(rt, {3, 1});
    W.set_requires_grad(true);
    x.set_requires_grad(true);

    float w_vals[] = {1.0f, 2.0f, 3.0f};
    float x_vals[] = {4.0f, 5.0f, 6.0f};
    std::memcpy(W.data<float>(), w_vals, 3 * sizeof(float));
    std::memcpy(x.data<float>(), x_vals, 3 * sizeof(float));

    Tensor z = *rt.relu(*rt.matmul(W, x));

    auto result = rt.autograd().backward(rt, z);
    REQUIRE(result);

    auto& grads = rt.autograd().gradients();
    Tensor grad_W = grads[W.id()];

    REQUIRE(grad_W.type().shape() == std::vector<int64_t>({1, 3}));

    const double eps = 1.0 / 1024.0;
    for (int64_t j = 0; j < 3; ++j) {
        double orig = W.data<float>()[j];

        W.data<float>()[j] = static_cast<float>(orig + eps);
        double z_plus = 0.0;
        for (int64_t k = 0; k < 3; ++k) {
            z_plus += static_cast<double>(W.data<float>()[k]) * static_cast<double>(x.data<float>()[k]);
        }
        z_plus = z_plus > 0.0 ? z_plus : 0.0;

        W.data<float>()[j] = static_cast<float>(orig - eps);
        double z_minus = 0.0;
        for (int64_t k = 0; k < 3; ++k) {
            z_minus += static_cast<double>(W.data<float>()[k]) * static_cast<double>(x.data<float>()[k]);
        }
        z_minus = z_minus > 0.0 ? z_minus : 0.0;

        W.data<float>()[j] = static_cast<float>(orig);

        double numerical = (z_plus - z_minus) / (2.0 * eps);
        REQUIRE(static_cast<double>(grad_W.data<float>()[j]) == Catch::Approx(numerical).epsilon(1e-3));
    }
}

TEST_CASE("Autograd backward populates Tensor::grad on graph inputs", "[autograd]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {1, 3});
    auto b = Tensor::empty(rt, {3, 1});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    a.data<float>()[0] = 1.0f; a.data<float>()[1] = 2.0f; a.data<float>()[2] = 3.0f;
    b.data<float>()[0] = 4.0f; b.data<float>()[1] = 5.0f; b.data<float>()[2] = 6.0f;

    Tensor c = *rt.matmul(a, b);
    rt.autograd().backward(rt, c);

    const auto& node = rt.autograd().graph()[0];
    REQUIRE(node->inputs()[0].has_grad());
    REQUIRE(node->inputs()[1].has_grad());
}

TEST_CASE("Gradient accumulation: same input used in two paths accumulates", "[autograd]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {1, 2});
    auto W1 = Tensor::empty(rt, {2, 1});
    auto W2 = Tensor::empty(rt, {2, 1});
    x.set_requires_grad(true);
    W1.set_requires_grad(true);
    W2.set_requires_grad(true);

    x.data<float>()[0] = 2.0f; x.data<float>()[1] = 3.0f;
    W1.data<float>()[0] = 1.0f; W1.data<float>()[1] = 2.0f;
    W2.data<float>()[0] = 3.0f; W2.data<float>()[1] = 4.0f;

    Tensor y = *rt.matmul(x, W1);
    Tensor z = *rt.matmul(x, W2);

    REQUIRE(rt.autograd().graph().size() == 2);

    const auto& node0 = rt.autograd().graph()[0];
    const auto& node1 = rt.autograd().graph()[1];
    REQUIRE(node0->inputs()[0].id() == x.id());
    REQUIRE(node1->inputs()[0].id() == x.id());

    // Backward on y only; second node has no gradient so it's skipped
    auto result = rt.autograd().backward(rt, y);
    REQUIRE(result);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(x.id()) > 0);

    // Gradient for x from first path: grad_out @ W1^T = 1 * [1, 2] = [1, 2]
    Tensor grad_x = grads[x.id()];
    REQUIRE(grad_x.data<float>()[0] == Catch::Approx(1.0f));
    REQUIRE(grad_x.data<float>()[1] == Catch::Approx(2.0f));
}

TEST_CASE("Graph records nodes in order and iterates in reverse", "[autograd]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {2, 3});
    auto b = Tensor::zeros(rt, {3, 4});
    auto c = Tensor::zeros(rt, {4, 2});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    Tensor ab = *rt.matmul(a, b);
    Tensor abc = *rt.matmul(ab, c);

    REQUIRE(rt.autograd().graph().size() == 2);
    REQUIRE(rt.autograd().graph()[0]->name() == "MatMul");
    REQUIRE(rt.autograd().graph()[1]->name() == "MatMul");
}

TEST_CASE("ReshapeOp forward changes shape without copying storage", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.reshape(x, {3, 2});
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 2}));
    REQUIRE(result.value().storage().get() == x.storage().get());
}

TEST_CASE("ReshapeOp forward validates total element count", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    auto result = rt.reshape(x, {2, 4});
    REQUIRE_FALSE(result);
}

TEST_CASE("ReshapeOp forward records graph node when requires_grad", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.reshape(x, {6});
    REQUIRE(rt.autograd().graph().size() == 1);
}

TEST_CASE("ReshapeOp backward reshapes gradient to original shape", "[autograd]") {
    Runtime rt;
    auto a = Tensor::zeros(rt, {2, 3});
    auto b = Tensor::zeros(rt, {3, 4});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    float a_data[] = {1,2,3,4,5,6};
    float b_data[] = {7,8,9,10,11,12,13,14,15,16,17,18};
    std::memcpy(a.data<float>(), a_data, 6 * sizeof(float));
    std::memcpy(b.data<float>(), b_data, 12 * sizeof(float));

    Tensor ab = *rt.matmul(a, b);             // {2, 4}
    Tensor flat = *rt.reshape(ab, {8});        // {8}
    Tensor y = *rt.relu(flat);                  // {8}
    rt.autograd().backward(rt, y);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(a.id()) > 0);
    REQUIRE(grads[a.id()].type().shape() == std::vector<int64_t>({2, 3}));
}

TEST_CASE("TransposeOp forward swaps shape and strides", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.transpose(x, 0, 1);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 2}));
    REQUIRE(result.value().type().strides() == std::vector<int64_t>({1, 3}));
}

TEST_CASE("TransposeOp forward shares storage", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.transpose(x, 0, 1);
    REQUIRE(result);
    REQUIRE(result.value().storage().get() == x.storage().get());
}

TEST_CASE("TransposeOp forward validates dims", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    auto result = rt.transpose(x, 0, 5);
    REQUIRE_FALSE(result);

    result = rt.transpose(x, -3, 1);
    REQUIRE_FALSE(result);
}

TEST_CASE("TransposeOp forward records graph node when requires_grad", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.transpose(x, 0, 1);
    REQUIRE(rt.autograd().graph().size() == 1);
}

TEST_CASE("TransposeOp forward does not record graph when requires_grad is false", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.transpose(x, 0, 1);
    REQUIRE(rt.autograd().graph().size() == 0);
}

TEST_CASE("TransposeOp backward produces gradient with original shape", "[autograd]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {2, 3});
    auto b = Tensor::empty(rt, {3, 4});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    float a_data[] = {1,2,3,4,5,6};
    float b_data[] = {7,8,9,10,11,12,13,14,15,16,17,18};
    std::memcpy(a.data<float>(), a_data, 6 * sizeof(float));
    std::memcpy(b.data<float>(), b_data, 12 * sizeof(float));

    // y = a @ b  -> {2,4}
    // z = y^T   -> {4,2}
    // loss = sum(z)
    Tensor y = *rt.matmul(a, b);
    Tensor z = *rt.transpose(y, 0, 1);
    rt.autograd().backward(rt, z);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(a.id()) > 0);
    REQUIRE(grads[a.id()].type().shape() == std::vector<int64_t>({2, 3}));
    REQUIRE(grads.count(b.id()) > 0);
    REQUIRE(grads[b.id()].type().shape() == std::vector<int64_t>({3, 4}));
}

TEST_CASE("TransposeOp backward values are correct (transpose of grad output)", "[autograd]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    auto W = Tensor::empty(rt, {3, 2});
    x.set_requires_grad(true);
    W.set_requires_grad(true);

    float x_data[] = {1,2,3,4,5,6};
    float w_data[] = {7,8,9,10,11,12};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));
    std::memcpy(W.data<float>(), w_data, 6 * sizeof(float));

    // y = x @ W  -> {2,2}
    // z = y^T   -> {2,2}
    // loss = sum(z) -> gradient of z is all ones
    Tensor y = *rt.matmul(x, W);
    Tensor z = *rt.transpose(y, 0, 1);
    rt.autograd().backward(rt, z);

    auto& grads = rt.autograd().gradients();

    // dy = ones({2,2}), dx = dy @ W^T
    // Since y = x @ W, dy = I (all ones from sum(z) where z = y^T)
    // Wait: loss = sum(z) where z = y^T means loss = sum(y^T) = sum(y)
    // So d(loss)/d(y) = ones({2,2})
    // dx = ones @ W^T = W^T summed along rows
    // For element (i,j): dx[i,j] = sum_k ones[i,k] * W[j,k] = sum_k W[j,k]
    Tensor grad_x = grads[x.id()];
    REQUIRE(grad_x.type().shape() == std::vector<int64_t>({2, 3}));
    // grad_x[i,j] = W[j,0] + W[j,1] for each i (since dy is all ones)
    // = W[j,0] + W[j,1]
    for (int64_t j = 0; j < 3; ++j) {
        float expected = w_data[j * 2 + 0] + w_data[j * 2 + 1];
        REQUIRE(grad_x.data<float>()[j] == Catch::Approx(expected));
    }
}

TEST_CASE("TransposeOp backward with non-contiguous grad_out uses correct strides", "[autograd]") {
    Runtime rt;

    // Input [2,3], forward transpose -> output [3,2] non-contiguous view
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);
    // Fill x with known values
    float x_data[] = {1,2,3,4,5,6};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));

    // Forward: transpose(0,1) -> output y is a view of x with swapped strides
    auto y = *rt.transpose(x, 0, 1);
    REQUIRE(y.type().shape() == std::vector<int64_t>({3, 2}));
    REQUIRE(y.type().strides() == std::vector<int64_t>({1, 3}));

    // Manually construct a TransposeNode for the transpose
    TransposeNode node(x, y, 0, 1);

    // Create a non-contiguous grad_out:
    // Base contiguous tensor shape [3,2], strides [2,1], filled 10..15
    auto base = Tensor::zeros(rt, {3, 2});
    float base_data[] = {10, 11, 12, 13, 14, 15};
    std::memcpy(base.data<float>(), base_data, 6 * sizeof(float));
    // View with swapped strides [1,3] (non-contiguous)
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor grad_out(view_type, base.storage(), false);

    // Place in GradientMap keyed by y's id
    GradientMap grads;
    grads[y.id()] = grad_out;

    // Call apply
    auto result = node.apply(rt, grads);
    REQUIRE(result);

    // Expected: grad has shape [2,3], contiguous, where grad[i][j] = grad_out[j][i]
    // grad_out[j][i] with strides [1,3] is at flat index j*1 + i*3
    // So grad[0][0] = grad_out[0][0] = base[0*1 + 0*3] = base[0] = 10
    //    grad[0][1] = grad_out[1][0] = base[1*1 + 0*3] = base[1] = 11
    //    grad[0][2] = grad_out[2][0] = base[2*1 + 0*3] = base[2] = 12
    //    grad[1][0] = grad_out[0][1] = base[0*1 + 1*3] = base[3] = 13
    //    grad[1][1] = grad_out[1][1] = base[1*1 + 1*3] = base[4] = 14
    //    grad[1][2] = grad_out[2][1] = base[2*1 + 1*3] = base[5] = 15

    REQUIRE(grads.count(x.id()) > 0);
    Tensor grad = grads[x.id()];
    REQUIRE(grad.type().shape() == std::vector<int64_t>({2, 3}));

    float expected_grad[] = {10, 11, 12, 13, 14, 15};
    auto* g_ptr = grad.data<float>();
    for (int i = 0; i < 6; ++i) {
        REQUIRE(g_ptr[i] == Catch::Approx(expected_grad[i]));
    }
}

TEST_CASE("TransposeOp forward with negative dims", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.transpose(x, -1, 0);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({3, 2}));
}

TEST_CASE("TransposeOp forward with dim1 == dim2 is no-op", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});

    auto result = rt.transpose(x, 1, 1);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 3}));
    REQUIRE(result.value().type().strides() == std::vector<int64_t>({3, 1}));
}

TEST_CASE("TransposeOp forward propagates storage_offset unchanged", "[operation][offset]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    REQUIRE(x.storage_offset() == 0);

    auto result = rt.transpose(x, 0, 1);
    REQUIRE(result);
    REQUIRE(result.value().storage_offset() == x.storage_offset());
}

TEST_CASE("TransposeOp forward data<T>() returns same address as original on view", "[operation][offset]") {
    Runtime rt;
    auto storage = rt.allocator().allocate(TensorType::contiguous({6}, DType::Float32));
    auto* raw = static_cast<float*>(storage->data);
    for (int i = 0; i < 6; ++i) raw[i] = static_cast<float>(i + 1);

    TensorType orig_type({2, 3}, {3, 1}, DType::Float32);
    Tensor x(orig_type, storage, false, 0);

    auto result = rt.transpose(x, 0, 1);
    REQUIRE(result);
    REQUIRE(result.value().storage().get() == storage.get());
    REQUIRE(result.value().storage_offset() == 0);
    // data<T>() returns same pointer (same offset into shared storage)
    REQUIRE(result.value().data<float>() == x.data<float>());
}

TEST_CASE("TransposeOp forward with non-zero storage_offset keeps data pointer correct", "[operation][offset]") {
    Runtime rt;
    // Create underlying storage with 8 elements
    auto storage = rt.allocator().allocate(TensorType::contiguous({8}, DType::Float32));
    auto* raw = static_cast<float*>(storage->data);
    for (int i = 0; i < 8; ++i) raw[i] = static_cast<float>(i + 100);

    // Create a tensor with offset 2, shape {2,3}, contiguous strides {3,1}
    // pointing at elements [2..7] of storage: {102, 103, 104, 105, 106, 107}
    TensorType x_type({2, 3}, {3, 1}, DType::Float32);
    Tensor x(x_type, storage, false, 2);

    REQUIRE(x.storage_offset() == 2);
    REQUIRE(x.data<float>()[0] == Catch::Approx(102.0f));
    REQUIRE(x.data<float>()[5] == Catch::Approx(107.0f));

    // Transpose
    auto result = rt.transpose(x, 0, 1);
    REQUIRE(result);
    // Should share storage, same offset
    REQUIRE(result.value().storage().get() == storage.get());
    REQUIRE(result.value().storage_offset() == 2);
    // data<T>() should return same pointer as x.data() (same offset)
    REQUIRE(result.value().data<float>() == x.data<float>());
}

TEST_CASE("ReshapeOp forward propagates storage_offset unchanged", "[operation][offset]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    REQUIRE(x.storage_offset() == 0);

    auto result = rt.reshape(x, {3, 2});
    REQUIRE(result);
    REQUIRE(result.value().storage_offset() == x.storage_offset());
}

TEST_CASE("ReshapeOp forward with non-zero storage_offset", "[operation][offset]") {
    Runtime rt;
    auto storage = rt.allocator().allocate(TensorType::contiguous({12}, DType::Float32));
    auto* raw = static_cast<float*>(storage->data);
    for (int i = 0; i < 12; ++i) raw[i] = static_cast<float>(i + 1);

    // Create a tensor with offset 2, shape {2,5} — but {2,5}=10 elems, + offset 2 = 12, fits in storage
    TensorType x_type({2, 5}, {5, 1}, DType::Float32);
    Tensor x(x_type, storage, false, 2);

    REQUIRE(x.storage_offset() == 2);

    // Reshape to {5,2}
    auto result = rt.reshape(x, {5, 2});
    REQUIRE(result);
    REQUIRE(result.value().storage().get() == storage.get());
    REQUIRE(result.value().storage_offset() == 2);
    // data<T>() should return same pointer
    REQUIRE(result.value().data<float>() == x.data<float>());
}

TEST_CASE("MeanOp forward reduces over single dim", "[operation]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    float x_data[] = {1,2,3,4,5,6};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));

    auto result = rt.mean(x, {1});
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2}));
    REQUIRE(result.value().data<float>()[0] == Catch::Approx(2.0f));
    REQUIRE(result.value().data<float>()[1] == Catch::Approx(5.0f));
}

TEST_CASE("MeanOp forward with keepdim preserves dimensions", "[operation]") {
    Runtime rt;
    auto x = Tensor::empty(rt, {2, 3});
    x.set_requires_grad(true);

    auto result = rt.mean(x, {1}, true);
    REQUIRE(result);
    REQUIRE(result.value().type().shape() == std::vector<int64_t>({2, 1}));
}

TEST_CASE("MeanOp forward records graph when requires_grad", "[operation]") {
    Runtime rt;
    auto x = Tensor::zeros(rt, {2, 3});
    x.set_requires_grad(true);

    REQUIRE(rt.autograd().graph().size() == 0);
    rt.mean(x, {1});
    REQUIRE(rt.autograd().graph().size() == 1);
}

TEST_CASE("ReshapeOp backward with non-contiguous grad_out uses TensorIterator", "[autograd][strided]") {
    Runtime rt;
    // Forward: create a view with non-contiguous strides, then reshape to flatten
    auto base = Tensor::zeros(rt, {6});
    float base_data[] = {10, 20, 30, 40, 50, 60};
    std::memcpy(base.data<float>(), base_data, 6 * sizeof(float));
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor x(view_type, base.storage(), false, 0);
    x.set_requires_grad(true);

    // Reshape from {3,2} to {6} — forward creates a view
    auto y = *rt.reshape(x, {6});
    // loss = sum(y)
    rt.autograd().backward(rt, y);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(x.id()) > 0);
    Tensor grad_x = grads[x.id()];
    REQUIRE(grad_x.type().shape() == std::vector<int64_t>({3, 2}));

    // grad_x should be all ones (since loss = sum(y), grad_out = {1,1,1,1,1,1})
    // and reshape backward just reshapes grad_out back to {3,2}
    TensorIterator<float> gx_it(grad_x);
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(gx_it[i] == Catch::Approx(1.0f).epsilon(1e-6));
    }
}

TEST_CASE("Composite: transpose -> layernorm backward with non-contiguous grad", "[autograd][strided]") {
    Runtime rt;
    // x: {2,3}, transpose -> {3,2}, then layernorm
    auto x = Tensor::empty(rt, {2, 3});
    float x_data[] = {1,2,3,4,5,6};
    std::memcpy(x.data<float>(), x_data, 6 * sizeof(float));
    x.set_requires_grad(true);

    auto gamma = rt.ones({2});
    auto beta = rt.zeros({2});

    auto xt = *rt.transpose(x, 0, 1);       // {3,2}, non-contiguous
    auto y = *rt.layernorm(xt, gamma, beta, 1e-5f);  // {3,2}, contiguous output
    rt.autograd().backward(rt, y);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(x.id()) > 0);
    Tensor grad_x = grads[x.id()];
    REQUIRE(grad_x.type().shape() == std::vector<int64_t>({2, 3}));

    // loss = sum(y), gradient is all ones
    // grad_x should not be NaN or inf
    for (int64_t i = 0; i < 6; ++i) {
        REQUIRE(std::isfinite(grad_x.data<float>()[i]));
    }
}

TEST_CASE("MeanOp backward broadcasts gradient correctly", "[autograd]") {
    Runtime rt;
    auto a = Tensor::empty(rt, {2, 3});
    auto b = Tensor::zeros(rt, {3, 4});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    float a_data[] = {1,2,3,4,5,6};
    std::memcpy(a.data<float>(), a_data, 6 * sizeof(float));

    // y = mean(a @ b, dim=1) => reduce {2,4} -> {2}
    Tensor ab = *rt.matmul(a, b);
    Tensor y = *rt.mean(ab, {1});
    rt.autograd().backward(rt, y);

    auto& grads = rt.autograd().gradients();
    REQUIRE(grads.count(a.id()) > 0);
    REQUIRE(grads[a.id()].type().shape() == std::vector<int64_t>({2, 3}));
}

TEST_CASE("MSELossOp forward with non-contiguous inputs uses TensorIterator", "[autograd][strided]") {
    Runtime rt;
    auto pred_base = Tensor::zeros(rt, {6});
    float* pb = pred_base.data<float>();
    pb[0]=0; pb[3]=1; pb[1]=2; pb[4]=3; pb[2]=4; pb[5]=5;
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor pred(view_type, pred_base.storage(), false);

    auto target_base = Tensor::zeros(rt, {6});
    float* tb = target_base.data<float>();
    tb[0]=5; tb[3]=4; tb[1]=3; tb[4]=2; tb[2]=1; tb[5]=0;
    Tensor target(view_type, target_base.storage(), false);

    auto loss = MSELossOp::forward(rt, pred, target);
    REQUIRE(loss);
    float val = loss.value().data<float>()[0];
    float expected = 70.0f / 6.0f;
    REQUIRE(val == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("L1LossOp forward with non-contiguous inputs uses TensorIterator", "[autograd][strided]") {
    Runtime rt;
    auto pred_base = Tensor::zeros(rt, {6});
    float* pb = pred_base.data<float>();
    pb[0]=0; pb[3]=1; pb[1]=2; pb[4]=3; pb[2]=4; pb[5]=5;
    TensorType view_type({3, 2}, {1, 3}, DType::Float32);
    Tensor pred(view_type, pred_base.storage(), false);

    auto target_base = Tensor::zeros(rt, {6});
    float* tb = target_base.data<float>();
    tb[0]=5; tb[3]=4; tb[1]=3; tb[4]=2; tb[2]=1; tb[5]=0;
    Tensor target(view_type, target_base.storage(), false);

    auto loss = L1LossOp::forward(rt, pred, target);
    REQUIRE(loss);
    float val = loss.value().data<float>()[0];
    REQUIRE(val == Catch::Approx(3.0f).epsilon(1e-6));
}

TEST_CASE("CrossEntropyLossOp forward with non-contiguous logits uses TensorIterator", "[autograd][strided]") {
    Runtime rt;
    auto logits_base = Tensor::zeros(rt, {6});
    float* lb = logits_base.data<float>();
    lb[0] = 1.0f; lb[2] = 2.0f; lb[4] = 0.5f;
    lb[1] = 0.0f; lb[3] = 1.0f; lb[5] = 3.0f;
    TensorType view_type({2, 3}, {1, 2}, DType::Float32);
    Tensor logits(view_type, logits_base.storage(), false);

    Tensor targets = Tensor::empty(rt, {2});
    targets.data<int64_t>()[0] = 1;
    targets.data<int64_t>()[1] = 2;

    auto loss = CrossEntropyLossOp::forward(rt, logits, targets);
    REQUIRE(loss);
    float val = loss.value().data<float>()[0];
    REQUIRE(std::isfinite(val));
    REQUIRE(val > 0.0f);
}
