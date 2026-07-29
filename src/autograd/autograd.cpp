#include "axon/autograd/autograd.h"
#include "axon/backend/cpu_backend.h"
#include "axon/runtime/runtime.h"

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
