#include "axon/autograd/autograd.h"
#include "axon/backend/cpu_backend.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/mse.h"
#include "axon/runtime/runtime.h"
#include <cmath>

namespace axon {

// ── Graph ─────────────────────────────────────────────────────────────

void Graph::append(GraphNode node) {
    nodes_.push_back(std::move(node));
}

size_t Graph::size() const {
    return nodes_.size();
}

const GraphNode& Graph::operator[](size_t i) const {
    return nodes_[i];
}

GraphNode& Graph::operator[](size_t i) {
    return nodes_[i];
}

// ── MatMulOp ──────────────────────────────────────────────────────────

Expected<Tensor> MatMulOp::forward(Runtime& rt, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"MatMulOp: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"MatMulOp: inner dimension mismatch"};
    }

    auto M = a.type().shape()[0];
    auto N = b.type().shape()[1];
    auto out_type = TensorType::contiguous({M, N}, a.type().dtype());
    bool need_grad = a.requires_grad() || b.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::matmul(out, a, b));

    if (need_grad) {
        GraphNode node;
        node.op = OpType::MatMul;
        node.inputs = {a, b};
        node.output = out;
        node.runtime = &rt;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> MatMulOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& a = node.inputs[0];
    const Tensor& b = node.inputs[1];
    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];

    // da = grad_out @ b^T  (M x K)
    {
        auto da_type = TensorType::contiguous({M, K}, a.type().dtype());
        Tensor da(da_type, rt.allocator().allocate(da_type), false);
        auto* da_ptr = da.data<float>();
        auto* go_ptr = grad_out.data<const float>();
        auto* b_ptr = b.data<const float>();

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < K; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < N; ++k) {
                    sum += go_ptr[i * N + k] * b_ptr[j * N + k];
                }
                da_ptr[i * K + j] = sum;
            }
        }

        auto it = grads.find(a.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, da);
        } else {
            grads[a.id()] = da;
        }
    }

    // db = a^T @ grad_out  (K x N)
    {
        auto db_type = TensorType::contiguous({K, N}, b.type().dtype());
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        auto* db_ptr = db.data<float>();
        auto* a_ptr = a.data<const float>();
        auto* go_ptr = grad_out.data<const float>();

        for (int64_t i = 0; i < K; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < M; ++k) {
                    sum += a_ptr[k * K + i] * go_ptr[k * N + j];
                }
                db_ptr[i * N + j] = sum;
            }
        }

        auto it = grads.find(b.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[b.id()] = db;
        }
    }

    return {};
}

// ── ReLUOp ────────────────────────────────────────────────────────────

Expected<Tensor> ReLUOp::forward(Runtime& rt, const Tensor& x) {
    auto out_type = TensorType::contiguous(x.type().shape(), x.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), x.requires_grad());

    RETURN_IF_ERROR(cpu::relu(out, x));

    if (x.requires_grad()) {
        GraphNode node;
        node.op = OpType::ReLU;
        node.inputs = {x};
        node.output = out;
        node.runtime = &rt;
        node.op_data = x;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> ReLUOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& x = node.inputs[0];
    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto dx_type = TensorType::contiguous(x.type().shape(), x.type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    auto* dx_ptr = dx.data<float>();
    auto* go_ptr = grad_out.data<const float>();
    auto* x_ptr = x.data<const float>();
    auto n = x.type().numel();

    for (int64_t i = 0; i < n; ++i) {
        dx_ptr[i] = x_ptr[i] > 0.0f ? go_ptr[i] : 0.0f;
    }

    const Tensor& input = node.inputs[0];
    auto it = grads.find(input.id());
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input.id()] = dx;
    }

    return {};
}

// ── AddOp ──────────────────────────────────────────────────────────────

Expected<Tensor> AddOp::forward(Runtime& rt, const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.type().shape();
    const auto& b_shape = b.type().shape();

    // Determine output shape (broadcasting: b may be 1D bias for 2D a)
    std::vector<int64_t> out_shape;
    if (b_shape.size() == 1 && a_shape.size() == 2 && a_shape[1] == b_shape[0]) {
        out_shape = a_shape;
    } else if (a_shape != b_shape) {
        return Error{"AddOp: shape mismatch"};
    } else {
        out_shape = a_shape;
    }

    auto out_type = TensorType::contiguous(out_shape, a.type().dtype());
    bool need_grad = a.requires_grad() || b.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    auto* a_ptr = a.data<const float>();
    auto* out_ptr = out.data<float>();
    auto n = out.type().numel();

    if (b_shape.size() == 1 && a_shape.size() == 2 && a_shape[1] == b_shape[0]) {
        // Broadcast bias: a is (N, C), b is (C,), out is (N, C)
        auto* b_ptr = b.data<const float>();
        auto C = a_shape[1];
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = a_ptr[i] + b_ptr[i % C];
        }
    } else {
        RETURN_IF_ERROR(cpu::add(out, a, b));
    }

    if (need_grad) {
        GraphNode node;
        node.op = OpType::Add;
        node.inputs = {a, b};
        node.output = out;
        node.runtime = &rt;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> AddOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& a = node.inputs[0];
    const Tensor& b = node.inputs[1];
    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    // Gradient flows equally to both inputs
    {
        auto it = grads.find(a.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, grad_out);
        } else {
            grads[a.id()] = grad_out;
        }
    }

    {
        // For bias (1D), sum over batch dimension
        if (b.type().shape().size() == 1 && grad_out.type().shape().size() == 2) {
            auto M = grad_out.type().shape()[0];
            auto N = grad_out.type().shape()[1];
            auto db_type = TensorType::contiguous(b.type().shape(), b.type().dtype());
            Tensor db(db_type, rt.allocator().allocate(db_type), false);
            auto* db_ptr = db.data<float>();
            auto* go_ptr = grad_out.data<const float>();

            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < M; ++i) {
                    sum += go_ptr[i * N + j];
                }
                db_ptr[j] = sum;
            }

            auto it = grads.find(b.id());
            if (it != grads.end()) {
                cpu::add(it->second, it->second, db);
            } else {
                grads[b.id()] = db;
            }
        } else {
            auto it = grads.find(b.id());
            if (it != grads.end()) {
                cpu::add(it->second, it->second, grad_out);
            } else {
                grads[b.id()] = grad_out;
            }
        }
    }

    return {};
}

// ── CrossEntropyLossOp ────────────────────────────────────────────────

Expected<Tensor> CrossEntropyLossOp::forward(Runtime& rt, const Tensor& logits, const Tensor& targets) {
    const auto& shape = logits.type().shape();
    if (shape.size() != 2) {
        return Error{"CrossEntropyLossOp: logits must be 2D"};
    }

    auto N = shape[0];
    auto C = shape[1];

    // Allocate output: scalar loss {1}
    auto loss_type = TensorType::contiguous({1}, logits.type().dtype());
    bool need_grad = logits.requires_grad();
    auto loss = Tensor(loss_type, rt.allocator().allocate(loss_type), need_grad);

    // Compute log_softmax in a temp buffer and NLL loss
    auto ls_type = TensorType::contiguous({N, C}, logits.type().dtype());
    Tensor log_softmax_out(ls_type, rt.allocator().allocate(ls_type), false);
    RETURN_IF_ERROR(cpu::log_softmax(log_softmax_out, logits));

    auto* ls_ptr = log_softmax_out.data<const float>();
    auto* t_ptr = targets.data<const int64_t>();
    float loss_val = 0.0f;

    for (int64_t i = 0; i < N; ++i) {
        loss_val -= ls_ptr[i * C + t_ptr[i]];
    }
    loss.data<float>()[0] = loss_val / static_cast<float>(N);

    if (need_grad) {
        GraphNode node;
        node.op = OpType::CrossEntropyLoss;
        node.inputs = {logits, targets};
        node.output = loss;
        node.runtime = &rt;
        // Store log_softmax output as op_data for backward
        node.op_data = log_softmax_out;
        rt.autograd().graph().append(std::move(node));
    }

    return loss;
}

Expected<void> CrossEntropyLossOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& logits = node.inputs[0];
    const Tensor& targets = node.inputs[1];
    const Tensor& log_softmax_out = node.op_data;

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) {
        return {};
    }

    auto N = logits.type().shape()[0];
    auto C = logits.type().shape()[1];
    float inv_N = 1.0f / static_cast<float>(N);

    // d_logits = (softmax - one_hot(target)) / N
    auto dlogits_type = TensorType::contiguous({N, C}, logits.type().dtype());
    Tensor dlogits(dlogits_type, rt.allocator().allocate(dlogits_type), false);
    auto* d_ptr = dlogits.data<float>();
    auto* ls_ptr = log_softmax_out.data<const float>();
    auto* t_ptr = targets.data<const int64_t>();

    for (int64_t i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (int64_t j = 0; j < C; ++j) {
            float sm = std::exp(ls_ptr[i * C + j]);
            d_ptr[i * C + j] = (sm - (j == t_ptr[i] ? 1.0f : 0.0f)) * inv_N;
        }
    }

    auto it = grads.find(logits.id());
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dlogits);
    } else {
        grads[logits.id()] = dlogits;
    }

    return {};
}

// ── MSELossOp ─────────────────────────────────────────────────────────

Expected<Tensor> MSELossOp::forward(Runtime& rt, const Tensor& pred, const Tensor& target) {
    if (pred.type().shape() != target.type().shape()) {
        return Error{"MSELossOp: shape mismatch between pred and target"};
    }

    auto loss_type = TensorType::contiguous({1}, pred.type().dtype());
    bool need_grad = pred.requires_grad();
    auto loss = Tensor(loss_type, rt.allocator().allocate(loss_type), need_grad);

    auto* p_ptr = pred.data<const float>();
    auto* t_ptr = target.data<const float>();
    auto n = pred.type().numel();

    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float diff = p_ptr[i] - t_ptr[i];
        sum += diff * diff;
    }
    loss.data<float>()[0] = sum / static_cast<float>(n);

    if (need_grad) {
        GraphNode node;
        node.op = OpType::MSE;
        node.inputs = {pred, target};
        node.output = loss;
        node.runtime = &rt;
        rt.autograd().graph().append(std::move(node));
    }

    return loss;
}

Expected<void> MSELossOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& pred = node.inputs[0];
    const Tensor& target = node.inputs[1];

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) {
        return {};
    }

    auto n = pred.type().numel();
    float inv_N = 2.0f / static_cast<float>(n);

    auto dp_type = TensorType::contiguous(pred.type().shape(), pred.type().dtype());
    Tensor dp(dp_type, rt.allocator().allocate(dp_type), false);
    auto* dp_ptr = dp.data<float>();
    auto* p_ptr = pred.data<const float>();
    auto* t_ptr = target.data<const float>();

    for (int64_t i = 0; i < n; ++i) {
        dp_ptr[i] = (p_ptr[i] - t_ptr[i]) * inv_N;
    }

    auto it = grads.find(pred.id());
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dp);
    } else {
        grads[pred.id()] = dp;
    }

    return {};
}

// ── Autograd ──────────────────────────────────────────────────────────

Expected<void> Autograd::backward(Runtime& runtime, const Tensor& loss) {
    grads_.clear();

    auto grad_loss_type = TensorType::contiguous(loss.type().shape(), loss.type().dtype());
    Tensor grad_loss(grad_loss_type, runtime.allocator().allocate(grad_loss_type), false);
    auto* ptr = grad_loss.data<float>();
    auto n = loss.type().numel();
    for (int64_t i = 0; i < n; ++i) {
        ptr[i] = 1.0f;
    }
    grads_[loss.id()] = grad_loss;

    for (int64_t i = static_cast<int64_t>(graph_.size()) - 1; i >= 0; --i) {
        const auto& node = graph_[i];
        switch (node.op) {
            case OpType::MatMul: {
                RETURN_IF_ERROR(MatMulOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::ReLU: {
                RETURN_IF_ERROR(ReLUOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::Add: {
                RETURN_IF_ERROR(AddOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::CrossEntropyLoss: {
                RETURN_IF_ERROR(CrossEntropyLossOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::MSE: {
                RETURN_IF_ERROR(MSELossOp::backward(*node.runtime, node, grads_));
                break;
            }
        }
    }

    for (size_t i = 0; i < graph_.size(); ++i) {
        auto& node = graph_[i];
        for (auto& input : node.inputs) {
            auto it = grads_.find(input.id());
            if (it != grads_.end()) {
                input.set_grad(it->second);
            }
        }
    }

    return {};
}

} // namespace axon
