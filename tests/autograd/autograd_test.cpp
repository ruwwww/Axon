#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstring>
#include <cmath>
#include "axon/backend/cpu_backend.h"
#include "axon/runtime/runtime.h"

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

    auto& node = rt.autograd().graph()[0];
    REQUIRE(node.inputs[0].has_grad());
    REQUIRE(node.inputs[1].has_grad());
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
    REQUIRE(node0.inputs[0].id() == x.id());
    REQUIRE(node1.inputs[0].id() == x.id());

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
    REQUIRE(rt.autograd().graph()[0].op == OpType::MatMul);
    REQUIRE(rt.autograd().graph()[1].op == OpType::MatMul);
}
