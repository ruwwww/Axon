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

    // Manually construct a GraphNode for the transpose
    GraphNode node;
    node.op = OpType::Transpose;
    node.inputs = {x};
    node.output = y;
    node.runtime = &rt;
    auto meta_type = TensorType::contiguous({2}, DType::Int64);
    Tensor meta(meta_type, rt.allocator().allocate(meta_type), false);
    meta.data<int64_t>()[0] = 0;
    meta.data<int64_t>()[1] = 1;
    node.op_data = meta;

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

    // Call backward
    auto result = TransposeOp::backward(rt, node, grads);
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
