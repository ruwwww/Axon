#include "axon/autograd/autograd.h"
#include "axon/autograd/nodes.h"
#include "axon/backend/cpu_backend.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/mse.h"
#include "axon/nn/l1_loss.h"
#include "axon/runtime/runtime.h"
#include "axon/tensor/tensor_iterator.h"
#include <cstring>
#include <cmath>
#include <limits>

namespace axon {

// ── Graph ─────────────────────────────────────────────────────────────

void Graph::append(std::shared_ptr<Node> node) {
    nodes_.push_back(std::move(node));
}

size_t Graph::size() const {
    return nodes_.size();
}

const std::shared_ptr<Node>& Graph::operator[](size_t i) const {
    return nodes_[i];
}

std::shared_ptr<Node>& Graph::operator[](size_t i) {
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
        rt.autograd().graph().append(std::make_shared<MatMulNode>(a, b, out));
    }

    return out;
}

Expected<void> MatMulNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto M = inputs_[0].type().shape()[0];
    auto K = inputs_[0].type().shape()[1];
    auto N = inputs_[1].type().shape()[1];

    // da = grad_out @ b^T  (M x K)
    {
        auto da_type = TensorType::contiguous({M, K}, inputs_[0].type().dtype());
        Tensor da(da_type, rt.allocator().allocate(da_type), false);
        auto* da_ptr = da.data<float>();

        TensorIterator<const float> go_it(grad_out);
        TensorIterator<const float> b_it(inputs_[1]);

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < K; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < N; ++k) {
                    sum += go_it[i * N + k] * b_it[j * N + k];
                }
                da_ptr[i * K + j] = sum;
            }
        }

        auto it = grads.find(input_ids_[0]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, da);
        } else {
            grads[input_ids_[0]] = da;
        }
    }

    // db = a^T @ grad_out  (K x N)
    {
        auto db_type = TensorType::contiguous({K, N}, inputs_[1].type().dtype());
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        auto* db_ptr = db.data<float>();

        TensorIterator<const float> a_it(inputs_[0]);
        TensorIterator<const float> go_it(grad_out);

        for (int64_t i = 0; i < K; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < M; ++k) {
                    sum += a_it[k * K + i] * go_it[k * N + j];
                }
                db_ptr[i * N + j] = sum;
            }
        }

        auto it = grads.find(input_ids_[1]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[input_ids_[1]] = db;
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
        rt.autograd().graph().append(std::make_shared<ReLUNode>(x, out));
    }

    return out;
}

Expected<void> ReLUNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto dx_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    TensorIterator<const float> x_it(inputs_[0]);
    TensorIterator<const float> go_it(grad_out);
    TensorIterator<float> dx_it(dx);
    auto n = inputs_[0].type().numel();

    for (int64_t i = 0; i < n; ++i) {
        dx_it[i] = x_it[i] > 0.0f ? go_it[i] : 0.0f;
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

// ── AddOp ──────────────────────────────────────────────────────────────

Expected<Tensor> AddOp::forward(Runtime& rt, const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.type().shape();
    const auto& b_shape = b.type().shape();

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
        auto* b_ptr = b.data<const float>();
        auto C = a_shape[1];
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = a_ptr[i] + b_ptr[i % C];
        }
    } else {
        RETURN_IF_ERROR(cpu::add(out, a, b));
    }

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<AddNode>(a, b, out));
    }

    return out;
}

Expected<void> AddNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, grad_out);
    } else {
        grads[input_ids_[0]] = grad_out;
    }

    {
        if (inputs_[1].type().shape().size() == 1 && grad_out.type().shape().size() == 2) {
            auto M = grad_out.type().shape()[0];
            auto N = grad_out.type().shape()[1];
            auto db_type = TensorType::contiguous(inputs_[1].type().shape(), inputs_[1].type().dtype());
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

            auto it = grads.find(input_ids_[1]);
            if (it != grads.end()) {
                cpu::add(it->second, it->second, db);
            } else {
                grads[input_ids_[1]] = db;
            }
        } else {
            auto it = grads.find(input_ids_[1]);
            if (it != grads.end()) {
                cpu::add(it->second, it->second, grad_out);
            } else {
                grads[input_ids_[1]] = grad_out;
            }
        }
    }

    return {};
}

// ── SubOp ──────────────────────────────────────────────────────────────

Expected<Tensor> SubOp::forward(Runtime& rt, const Tensor& a, const Tensor& b) {
    if (a.type().shape() != b.type().shape()) {
        return Error{"SubOp: shape mismatch"};
    }

    auto out_type = TensorType::contiguous(a.type().shape(), a.type().dtype());
    bool need_grad = a.requires_grad() || b.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::sub(out, a, b));

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<SubNode>(a, b, out));
    }

    return out;
}

Expected<void> SubNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    {
        auto dx_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
        Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
        TensorIterator<const float> go_it(grad_out);
        TensorIterator<float> dx_it(dx);
        auto n = grad_out.type().numel();
        for (int64_t i = 0; i < n; ++i) {
            dx_it[i] = go_it[i];
        }
        auto it = grads.find(input_ids_[0]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dx);
        } else {
            grads[input_ids_[0]] = dx;
        }
    }

    {
        auto it = grads.find(input_ids_[1]);
        if (it != grads.end()) {
            auto neg_grad_type = TensorType::contiguous(inputs_[1].type().shape(), inputs_[1].type().dtype());
            Tensor neg_grad(neg_grad_type, rt.allocator().allocate(neg_grad_type), false);
            TensorIterator<const float> go_it(grad_out);
            TensorIterator<float> ng_it(neg_grad);
            auto n = grad_out.type().numel();
            for (int64_t i = 0; i < n; ++i) {
                ng_it[i] = -go_it[i];
            }
            cpu::add(it->second, it->second, neg_grad);
        } else {
            auto neg_grad_type = TensorType::contiguous(inputs_[1].type().shape(), inputs_[1].type().dtype());
            Tensor neg_grad(neg_grad_type, rt.allocator().allocate(neg_grad_type), false);
            TensorIterator<const float> go_it(grad_out);
            TensorIterator<float> ng_it(neg_grad);
            auto n = grad_out.type().numel();
            for (int64_t i = 0; i < n; ++i) {
                ng_it[i] = -go_it[i];
            }
            grads[input_ids_[1]] = neg_grad;
        }
    }

    return {};
}

// ── MulScalarOp ────────────────────────────────────────────────────────

Expected<Tensor> MulScalarOp::forward(Runtime& rt, const Tensor& x, float scalar) {
    auto out_type = TensorType::contiguous(x.type().shape(), x.type().dtype());
    bool need_grad = x.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::mul_scalar(out, x, scalar));

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<MulScalarNode>(x, out, scalar));
    }

    return out;
}

Expected<void> MulScalarNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto dx_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    TensorIterator<const float> go_it(grad_out);
    TensorIterator<float> dx_it(dx);
    auto n = grad_out.type().numel();
    float s = scalar_;

    for (int64_t i = 0; i < n; ++i) {
        dx_it[i] = go_it[i] * s;
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

// ── DivScalarOp ────────────────────────────────────────────────────────

Expected<Tensor> DivScalarOp::forward(Runtime& rt, const Tensor& x, float scalar) {
    auto out_type = TensorType::contiguous(x.type().shape(), x.type().dtype());
    bool need_grad = x.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::div_scalar(out, x, scalar));

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<DivScalarNode>(x, out, scalar));
    }

    return out;
}

Expected<void> DivScalarNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto dx_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    TensorIterator<const float> go_it(grad_out);
    TensorIterator<float> dx_it(dx);
    auto n = grad_out.type().numel();
    float inv_s = 1.0f / scalar_;

    for (int64_t i = 0; i < n; ++i) {
        dx_it[i] = go_it[i] * inv_s;
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

Expected<Tensor> CrossEntropyLossOp::forward(Runtime& rt, const Tensor& logits, const Tensor& targets) {
    const auto& shape = logits.type().shape();
    if (shape.size() != 2) {
        return Error{"CrossEntropyLossOp: logits must be 2D"};
    }

    auto N = shape[0];
    auto C = shape[1];

    auto loss_type = TensorType::contiguous({1}, logits.type().dtype());
    bool need_grad = logits.requires_grad();
    auto loss = Tensor(loss_type, rt.allocator().allocate(loss_type), need_grad);

    auto ls_type = TensorType::contiguous({N, C}, logits.type().dtype());
    Tensor log_softmax_out(ls_type, rt.allocator().allocate(ls_type), false);
    RETURN_IF_ERROR(cpu::log_softmax(log_softmax_out, logits));

    TensorIterator<const float> ls_it(log_softmax_out);
    TensorIterator<const int64_t> t_it(targets);
    float loss_val = 0.0f;

    for (int64_t i = 0; i < N; ++i) {
        loss_val -= ls_it[i * C + t_it[i]];
    }
    loss.data<float>()[0] = loss_val / static_cast<float>(N);

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<CrossEntropyLossNode>(logits, targets, loss, log_softmax_out));
    }

    return loss;
}

Expected<void> CrossEntropyLossNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }

    auto N = inputs_[0].type().shape()[0];
    auto C = inputs_[0].type().shape()[1];
    float inv_N = 1.0f / static_cast<float>(N);

    auto dlogits_type = TensorType::contiguous({N, C}, inputs_[0].type().dtype());
    Tensor dlogits(dlogits_type, rt.allocator().allocate(dlogits_type), false);

    TensorIterator<const float> ls_it(log_softmax_out_);
    TensorIterator<const int64_t> t_it(inputs_[1]);
    TensorIterator<float> d_it(dlogits);

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < C; ++j) {
            float sm = std::exp(ls_it[i * C + j]);
            d_it[i * C + j] = (sm - (j == t_it[i] ? 1.0f : 0.0f)) * inv_N;
        }
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dlogits);
    } else {
        grads[input_ids_[0]] = dlogits;
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

    TensorIterator<const float> p_it(pred);
    TensorIterator<const float> t_it(target);
    auto n = pred.type().numel();

    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float diff = p_it[i] - t_it[i];
        sum += diff * diff;
    }
    loss.data<float>()[0] = sum / static_cast<float>(n);

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<MSELossNode>(pred, target, loss));
    }

    return loss;
}

Expected<void> MSELossNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }

    auto n = inputs_[0].type().numel();
    float scale = 2.0f / static_cast<float>(n);

    auto dpred_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
    Tensor dpred(dpred_type, rt.allocator().allocate(dpred_type), false);

    TensorIterator<const float> p_it(inputs_[0]);
    TensorIterator<const float> t_it(inputs_[1]);
    TensorIterator<float> d_it(dpred);

    for (int64_t i = 0; i < n; ++i) {
        d_it[i] = scale * (p_it[i] - t_it[i]);
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dpred);
    } else {
        grads[input_ids_[0]] = dpred;
    }

    return {};
}

// ── L1LossOp ──────────────────────────────────────────────────────────

Expected<Tensor> L1LossOp::forward(Runtime& rt, const Tensor& pred, const Tensor& target) {
    if (pred.type().shape() != target.type().shape()) {
        return Error{"L1LossOp: shape mismatch between pred and target"};
    }

    auto loss_type = TensorType::contiguous({1}, pred.type().dtype());
    bool need_grad = pred.requires_grad();
    auto loss = Tensor(loss_type, rt.allocator().allocate(loss_type), need_grad);

    TensorIterator<const float> p_it(pred);
    TensorIterator<const float> t_it(target);
    auto n = pred.type().numel();

    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(p_it[i] - t_it[i]);
    }
    loss.data<float>()[0] = sum / static_cast<float>(n);

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<L1LossNode>(pred, target, loss));
    }

    return loss;
}

Expected<void> L1LossNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }

    auto n = inputs_[0].type().numel();
    float scale = 1.0f / static_cast<float>(n);

    auto dpred_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
    Tensor dpred(dpred_type, rt.allocator().allocate(dpred_type), false);

    TensorIterator<const float> p_it(inputs_[0]);
    TensorIterator<const float> t_it(inputs_[1]);
    TensorIterator<float> d_it(dpred);

    for (int64_t i = 0; i < n; ++i) {
        float diff = p_it[i] - t_it[i];
        if (diff > 0.0f) {
            d_it[i] = scale;
        } else if (diff < 0.0f) {
            d_it[i] = -scale;
        } else {
            d_it[i] = 0.0f;
        }
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dpred);
    } else {
        grads[input_ids_[0]] = dpred;
    }

    return {};
}

// ── Conv2DOp ───────────────────────────────────────────────────────────

Expected<Tensor> Conv2DOp::forward(Runtime& rt, const Tensor& input, const Tensor& weight,
                                    const Tensor& bias, int64_t stride, int64_t padding) {
    const auto& in_shape = input.type().shape();
    const auto& w_shape = weight.type().shape();

    if (in_shape.size() != 4 || w_shape.size() != 4) {
        return Error{"Conv2DOp: input must be 4D (N,C,H,W) and weight 4D (OC,IC,KH,KW)"};
    }

    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OC = w_shape[0], IC = w_shape[1], KH = w_shape[2], KW = w_shape[3];

    if (C != IC) return Error{"Conv2DOp: channel mismatch"};

    int64_t OH = (H + 2 * padding - KH) / stride + 1;
    int64_t OW = (W + 2 * padding - KW) / stride + 1;

    auto out_type = TensorType::contiguous({N, OC, OH, OW}, input.type().dtype());
    bool need_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::conv2d(out, input, weight, stride, padding));

    if (bias.defined() && bias.type().numel() > 0) {
        TensorIterator<float> o_it(out);
        TensorIterator<const float> b_it(bias);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc = 0; oc < OC; ++oc) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        o_it[n * OC * OH * OW + oc * OH * OW + oh * OW + ow] += b_it[oc];
                    }
                }
            }
        }
    }

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<Conv2DNode>(input, weight, bias, out, stride, padding));
    }

    return out;
}

Expected<void> Conv2DNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& in_shape = inputs_[0].type().shape();
    const auto& w_shape = inputs_[1].type().shape();
    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OC = w_shape[0], IC = w_shape[1], KH = w_shape[2], KW = w_shape[3];
    auto OH = grad_out.type().shape()[2], OW = grad_out.type().shape()[3];

    // d_input: conv transpose of grad_out with weight
    if (inputs_[0].requires_grad()) {
        auto di_type = TensorType::contiguous({N, C, H, W}, inputs_[0].type().dtype());
        Tensor di(di_type, rt.allocator().allocate(di_type), false);
        TensorIterator<float> di_it(di);
        TensorIterator<const float> go_it(grad_out);
        TensorIterator<const float> w_it(inputs_[1]);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        float sum = 0.0f;
                        for (int64_t oc = 0; oc < OC; ++oc) {
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    int64_t oh = (h + padding_ - kh);
                                    int64_t ow = (w + padding_ - kw);
                                    if (oh % stride_ == 0 && ow % stride_ == 0) {
                                        oh /= stride_;
                                        ow /= stride_;
                                        if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
                                            sum += go_it[n * OC * OH * OW + oc * OH * OW + oh * OW + ow]
                                                 * w_it[oc * IC * KH * KW + c * KH * KW + kh * KW + kw];
                                        }
                                    }
                                }
                            }
                        }
                        di_it[n * C * H * W + c * H * W + h * W + w] = sum;
                    }
                }
            }
        }

        auto it = grads.find(input_ids_[0]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, di);
        } else {
            grads[input_ids_[0]] = di;
        }
    }

    // d_weight
    if (inputs_[1].requires_grad()) {
        auto dw_type = TensorType::contiguous({OC, IC, KH, KW}, inputs_[1].type().dtype());
        Tensor dw(dw_type, rt.allocator().allocate(dw_type), false);
        TensorIterator<float> dw_it(dw);
        TensorIterator<const float> inp_it(inputs_[0]);
        TensorIterator<const float> go_it(grad_out);

        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t ic = 0; ic < IC; ++ic) {
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        float sum = 0.0f;
                        for (int64_t n = 0; n < N; ++n) {
                            for (int64_t oh = 0; oh < OH; ++oh) {
                                for (int64_t ow = 0; ow < OW; ++ow) {
                                    int64_t ih = oh * stride_ + kh - padding_;
                                    int64_t iw = ow * stride_ + kw - padding_;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        sum += inp_it[n * C * H * W + ic * H * W + ih * W + iw]
                                             * go_it[n * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                                    }
                                }
                            }
                        }
                        dw_it[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw] = sum;
                    }
                }
            }
        }

        auto it = grads.find(input_ids_[1]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dw);
        } else {
            grads[input_ids_[1]] = dw;
        }
    }

    // d_bias: sum grad_out over N, H, W per output channel
    if (inputs_[2].defined() && inputs_[2].requires_grad() && inputs_[2].type().numel() > 0) {
        auto db_type = TensorType::contiguous({OC}, DType::Float32);
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        TensorIterator<float> db_it(db);
        TensorIterator<const float> go_it(grad_out);

        for (int64_t oc = 0; oc < OC; ++oc) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        sum += go_it[n * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                    }
                }
            }
            db_it[oc] = sum;
        }

        auto it = grads.find(input_ids_[2]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[input_ids_[2]] = db;
        }
    }

    return {};
}

// ── MaxPool2dOp ────────────────────────────────────────────────────────

Expected<Tensor> MaxPool2dOp::forward(Runtime& rt, const Tensor& input, int64_t kernel, int64_t stride) {
    const auto& in_shape = input.type().shape();
    if (in_shape.size() != 4) {
        return Error{"MaxPool2dOp: input must be 4D (N,C,H,W)"};
    }

    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    int64_t OH = (H - kernel) / stride + 1;
    int64_t OW = (W - kernel) / stride + 1;

    auto out_type = TensorType::contiguous({N, C, OH, OW}, input.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), input.requires_grad());

    RETURN_IF_ERROR(cpu::maxpool2d(out, input, kernel, stride));

    if (input.requires_grad()) {
        rt.autograd().graph().append(std::make_shared<MaxPool2dNode>(input, out, kernel, stride));
    }

    return out;
}

Expected<void> MaxPool2dNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) return {};

    const auto& in_shape = inputs_[0].type().shape();
    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OH = output_.type().shape()[2], OW = output_.type().shape()[3];

    auto dx_type = TensorType::contiguous({N, C, H, W}, inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
    TensorIterator<float> dx_it(dx);
    Tensor grad_out = grad_it->second;
    TensorIterator<const float> inp_it(inputs_[0]);
    TensorIterator<const float> go_it(grad_out);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    int64_t argmax_h = 0, argmax_w = 0;
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t kh = 0; kh < kernel_; ++kh) {
                        for (int64_t kw = 0; kw < kernel_; ++kw) {
                            int64_t ih = oh * stride_ + kh;
                            int64_t iw = ow * stride_ + kw;
                            if (ih < H && iw < W) {
                                float val = inp_it[n * C * H * W + c * H * W + ih * W + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    argmax_h = kh;
                                    argmax_w = kw;
                                }
                            }
                        }
                    }
                    int64_t ih = oh * stride_ + argmax_h;
                    int64_t iw = ow * stride_ + argmax_w;
                    dx_it[n * C * H * W + c * H * W + ih * W + iw] += go_it[n * C * OH * OW + c * OH * OW + oh * OW + ow];
                }
            }
        }
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

// ── AvgPool2dOp ────────────────────────────────────────────────────────

Expected<Tensor> AvgPool2dOp::forward(Runtime& rt, const Tensor& input, int64_t kernel, int64_t stride) {
    const auto& in_shape = input.type().shape();
    if (in_shape.size() != 4) {
        return Error{"AvgPool2dOp: input must be 4D (N,C,H,W)"};
    }

    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    int64_t OH = (H - kernel) / stride + 1;
    int64_t OW = (W - kernel) / stride + 1;

    auto out_type = TensorType::contiguous({N, C, OH, OW}, input.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), input.requires_grad());

    RETURN_IF_ERROR(cpu::avgpool2d(out, input, kernel, stride));

    if (input.requires_grad()) {
        rt.autograd().graph().append(std::make_shared<AvgPool2dNode>(input, out, kernel, stride));
    }

    return out;
}

Expected<void> AvgPool2dNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) return {};

    const auto& in_shape = inputs_[0].type().shape();
    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OH = output_.type().shape()[2], OW = output_.type().shape()[3];

    auto dx_type = TensorType::contiguous({N, C, H, W}, inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
    TensorIterator<float> dx_it(dx);
    Tensor grad_out = grad_it->second;
    TensorIterator<const float> go_it(grad_out);

    float inv_k = 1.0f / static_cast<float>(kernel_ * kernel_);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float g = go_it[n * C * OH * OW + c * OH * OW + oh * OW + ow];
                    for (int64_t kh = 0; kh < kernel_; ++kh) {
                        for (int64_t kw = 0; kw < kernel_; ++kw) {
                            int64_t ih = oh * stride_ + kh;
                            int64_t iw = ow * stride_ + kw;
                            if (ih < H && iw < W) {
                                dx_it[n * C * H * W + c * H * W + ih * W + iw] += g * inv_k;
                            }
                        }
                    }
                }
            }
        }
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

// ── BatchNormOp ────────────────────────────────────────────────────────

Expected<Tensor> BatchNormOp::forward(Runtime& rt, const Tensor& input,
                                       const Tensor& gamma, const Tensor& beta,
                                       const Tensor& running_mean, const Tensor& running_var,
                                       float momentum, float epsilon, bool training) {
    auto out_type = TensorType::contiguous(input.type().shape(), input.type().dtype());
    bool need_grad = input.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::batchnorm(out, input, gamma, beta, running_mean, running_var, momentum, epsilon, training));

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<BatchNormNode>(
            input, gamma, beta, running_mean, running_var, out, momentum, epsilon, training));
    }

    return out;
}

Expected<void> BatchNormNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& in_shape = inputs_[0].type().shape();
    int64_t N = in_shape[0], C = in_shape[1];
    int64_t spatial = 1;
    for (size_t i = 2; i < in_shape.size(); ++i) spatial *= in_shape[i];
    int64_t num_elements = N * spatial;

    TensorIterator<const float> inp_it(inputs_[0]);
    TensorIterator<const float> g_it(inputs_[1]);
    TensorIterator<const float> go_it(grad_out);

    std::vector<float> mean(C, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t s = 0; s < spatial; ++s)
                sum += inp_it[n * C * spatial + c * spatial + s];
            mean[c] += sum;
        }
    for (int64_t c = 0; c < C; ++c) mean[c] /= static_cast<float>(num_elements);

    std::vector<float> var(C, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t s = 0; s < spatial; ++s) {
                float diff = inp_it[n * C * spatial + c * spatial + s] - mean[c];
                sum += diff * diff;
            }
            var[c] += sum;
        }
    for (int64_t c = 0; c < C; ++c) var[c] /= static_cast<float>(num_elements);

    std::vector<float> inv_std(C);
    for (int64_t c = 0; c < C; ++c)
        inv_std[c] = 1.0f / std::sqrt(var[c] + epsilon_);

    if (inputs_[1].requires_grad()) {
        auto dg_type = TensorType::contiguous({C}, DType::Float32);
        Tensor dg(dg_type, rt.allocator().allocate(dg_type), false);
        TensorIterator<float> dg_it(dg);

        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t s = 0; s < spatial; ++s) {
                    float x_hat = (inp_it[n * C * spatial + c * spatial + s] - mean[c]) * inv_std[c];
                    sum += go_it[n * C * spatial + c * spatial + s] * x_hat;
                }
            dg_it[c] = sum;
        }

        auto it = grads.find(input_ids_[1]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dg);
        } else {
            grads[input_ids_[1]] = dg;
        }
    }

    if (inputs_[2].requires_grad()) {
        auto db_type = TensorType::contiguous({C}, DType::Float32);
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        TensorIterator<float> db_it(db);

        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t s = 0; s < spatial; ++s)
                    sum += go_it[n * C * spatial + c * spatial + s];
            db_it[c] = sum;
        }

        auto it = grads.find(input_ids_[2]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[input_ids_[2]] = db;
        }
    }

    if (inputs_[0].requires_grad()) {
        auto di_type = TensorType::contiguous(in_shape, DType::Float32);
        Tensor di(di_type, rt.allocator().allocate(di_type), false);
        TensorIterator<float> di_it(di);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float inv_N = 1.0f / static_cast<float>(num_elements);
                float g_val = g_it[c];
                float istd = inv_std[c];
                float x_mean_sum = 0.0f;
                float go_sum = 0.0f;

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    float x_hat = (inp_it[idx] - mean[c]) * istd;
                    x_mean_sum += x_hat;
                    go_sum += go_it[idx];
                }

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    float x_hat = (inp_it[idx] - mean[c]) * istd;
                    di_it[idx] = g_val * istd * inv_N * (
                        static_cast<float>(num_elements) * go_it[idx] - go_sum - x_hat * x_mean_sum
                    );
                }
            }
        }

        auto it = grads.find(input_ids_[0]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, di);
        } else {
            grads[input_ids_[0]] = di;
        }
    }

    return {};
}

// ── LayerNormOp ────────────────────────────────────────────────────────

Expected<Tensor> LayerNormOp::forward(Runtime& rt, const Tensor& input,
                                       const Tensor& gamma, const Tensor& beta, float epsilon) {
    auto out_type = TensorType::contiguous(input.type().shape(), input.type().dtype());
    bool need_grad = input.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::layernorm(out, input, gamma, beta, epsilon));

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<LayerNormNode>(input, gamma, beta, out, epsilon));
    }

    return out;
}

Expected<void> LayerNormNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& in_shape = inputs_[0].type().shape();
    auto N = in_shape[0];
    int64_t D = 1;
    for (size_t i = 1; i < in_shape.size(); ++i) D *= in_shape[i];

    TensorIterator<const float> inp_it(inputs_[0]);
    TensorIterator<const float> g_it(inputs_[1]);
    TensorIterator<const float> go_it(grad_out);

    std::vector<float> mean(N), var(N), inv_std(N);
    for (int64_t n = 0; n < N; ++n) {
        float m = 0.0f;
        for (int64_t d = 0; d < D; ++d) m += inp_it[n * D + d];
        mean[n] = m / static_cast<float>(D);

        float v = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float diff = inp_it[n * D + d] - mean[n];
            v += diff * diff;
        }
        var[n] = v / static_cast<float>(D);
        inv_std[n] = 1.0f / std::sqrt(var[n] + epsilon_);
    }

    if (inputs_[1].requires_grad()) {
        auto dg_type = TensorType::contiguous({D}, DType::Float32);
        Tensor dg(dg_type, rt.allocator().allocate(dg_type), false);
        TensorIterator<float> dg_it(dg);

        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                float x_hat = (inp_it[n * D + d] - mean[n]) * inv_std[n];
                sum += go_it[n * D + d] * x_hat;
            }
            dg_it[d] = sum;
        }

        auto it = grads.find(input_ids_[1]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dg);
        } else {
            grads[input_ids_[1]] = dg;
        }
    }

    if (inputs_[2].requires_grad()) {
        auto db_type = TensorType::contiguous({D}, DType::Float32);
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        TensorIterator<float> db_it(db);

        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) sum += go_it[n * D + d];
            db_it[d] = sum;
        }

        auto it = grads.find(input_ids_[2]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[input_ids_[2]] = db;
        }
    }

    if (inputs_[0].requires_grad()) {
        auto di_type = TensorType::contiguous(in_shape, DType::Float32);
        Tensor di(di_type, rt.allocator().allocate(di_type), false);
        TensorIterator<float> di_it(di);

        for (int64_t n = 0; n < N; ++n) {
            float inv_D = 1.0f / static_cast<float>(D);
            float istd = inv_std[n];

            float sum1 = 0.0f, sum2 = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                float x_hat = (inp_it[n * D + d] - mean[n]) * istd;
                sum1 += go_it[n * D + d] * g_it[d];
                sum2 += go_it[n * D + d] * g_it[d] * x_hat;
            }

            for (int64_t d = 0; d < D; ++d) {
                float x_hat = (inp_it[n * D + d] - mean[n]) * istd;
                di_it[n * D + d] = istd * (
                    go_it[n * D + d] * g_it[d] - inv_D * sum1 - x_hat * inv_D * sum2
                );
            }
        }

        auto it = grads.find(input_ids_[0]);
        if (it != grads.end()) {
            cpu::add(it->second, it->second, di);
        } else {
            grads[input_ids_[0]] = di;
        }
    }

    return {};
}

// ── GELUOp ────────────────────────────────────────────────────────────

Expected<Tensor> GELUOp::forward(Runtime& rt, const Tensor& x) {
    auto out_type = TensorType::contiguous(x.type().shape(), x.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), x.requires_grad());

    RETURN_IF_ERROR(cpu::gelu(out, x));

    if (x.requires_grad()) {
        rt.autograd().graph().append(std::make_shared<GELUNode>(x, out));
    }

    return out;
}

Expected<void> GELUNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto dx_type = TensorType::contiguous(inputs_[0].type().shape(), inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);

    TensorIterator<const float> x_it(inputs_[0]);
    TensorIterator<const float> go_it(grad_out);
    TensorIterator<float> dx_it(dx);
    auto n = inputs_[0].type().numel();
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

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

// ── ReshapeOp ─────────────────────────────────────────────────────────

Expected<Tensor> ReshapeOp::forward(Runtime& rt, const Tensor& x, const std::vector<int64_t>& new_shape) {
    int64_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != x.type().numel()) {
        return Error{"ReshapeOp: new shape has different number of elements"};
    }

    auto new_type = TensorType::contiguous(new_shape, x.type().dtype());
    bool need_grad = x.requires_grad();
    auto out = Tensor(new_type, x.storage(), need_grad, x.storage_offset());

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<ReshapeNode>(x, out));
    }

    return out;
}

Expected<void> ReshapeNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    auto grad_type = TensorType::contiguous(inputs_[0].type().shape(), grad_out.type().dtype());
    Tensor grad(grad_type, rt.allocator().allocate(grad_type), false);

    TensorIterator<float> grad_dst(grad);
    TensorIterator<const float> go_src(grad_out);
    auto n = grad_out.type().numel();
    for (int64_t i = 0; i < n; ++i) {
        grad_dst[i] = go_src[i];
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, grad);
    } else {
        grads[input_ids_[0]] = grad;
    }

    return {};
}

// ── MeanOp ─────────────────────────────────────────────────────────────

Expected<Tensor> MeanOp::forward(Runtime& rt, const Tensor& x, const std::vector<int64_t>& dims, bool keepdim) {
    const auto& shape = x.type().shape();
    auto ndim = shape.size();

    std::vector<bool> is_reduced(ndim, false);
    for (auto d : dims) {
        if (d < 0 || static_cast<size_t>(d) >= ndim)
            return Error{"MeanOp: dim out of range"};
        is_reduced[d] = true;
    }

    std::vector<int64_t> out_shape;
    for (size_t d = 0; d < ndim; ++d) {
        if (is_reduced[d]) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[d]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto out_type = TensorType::contiguous(out_shape, x.type().dtype());
    bool need_grad = x.requires_grad();
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), need_grad);

    RETURN_IF_ERROR(cpu::reduce_mean(out, x, dims));

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<MeanNode>(x, out, dims, keepdim));
    }

    return out;
}

Expected<void> MeanNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& orig_shape = inputs_[0].type().shape();
    auto ndim = orig_shape.size();

    int64_t reduction_size = 1;
    for (auto d : dims_) reduction_size *= orig_shape[d];

    auto dx_type = TensorType::contiguous(orig_shape, inputs_[0].type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
    TensorIterator<float> dx_it(dx);
    TensorIterator<const float> go_it(grad_out);
    auto numel = inputs_[0].type().numel();

    std::vector<bool> is_reduced(ndim, false);
    for (auto d : dims_) is_reduced[d] = true;

    float inv = 1.0f / static_cast<float>(reduction_size);
    std::vector<int64_t> idx(ndim, 0);
    for (int64_t flat = 0; flat < numel; ++flat) {
        int64_t tmp = flat;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            idx[d] = tmp % orig_shape[d];
            tmp /= orig_shape[d];
        }
        int64_t out_flat = 0;
        for (size_t d = 0; d < ndim; ++d) {
            if (!is_reduced[d]) {
                out_flat = out_flat * orig_shape[d] + idx[d];
            }
        }
        dx_it[flat] = go_it[out_flat] * inv;
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
    }

    return {};
}

// ── TransposeOp ───────────────────────────────────────────────────────

Expected<Tensor> TransposeOp::forward(Runtime& rt, const Tensor& x, int64_t dim1, int64_t dim2) {
    auto ndim = static_cast<int64_t>(x.type().shape().size());
    if (dim1 < 0) dim1 += ndim;
    if (dim2 < 0) dim2 += ndim;
    if (dim1 < 0 || dim1 >= ndim || dim2 < 0 || dim2 >= ndim) {
        return Error{"TransposeOp: dim out of range"};
    }

    auto swapped_shape = x.type().shape();
    auto swapped_strides = x.type().strides();
    std::swap(swapped_shape[dim1], swapped_shape[dim2]);
    std::swap(swapped_strides[dim1], swapped_strides[dim2]);

    TensorType out_type(swapped_shape, swapped_strides, x.type().dtype());
    bool need_grad = x.requires_grad();
    auto out = Tensor(out_type, x.storage(), need_grad, x.storage_offset());

    if (need_grad) {
        rt.autograd().graph().append(std::make_shared<TransposeNode>(x, out, dim1, dim2));
    }

    return out;
}

Expected<void> TransposeNode::apply(Runtime& rt, GradientMap& grads) {
    auto grad_it = grads.find(output_.id());
    if (grad_it == grads.end()) {
        return {};
    }
    Tensor grad_out = grad_it->second;

    const auto& in_shape = inputs_[0].type().shape();

    // Create a transposed view of grad_out to undo the forward transpose
    auto grad_shape = grad_out.type().shape();
    auto grad_strides = grad_out.type().strides();
    std::swap(grad_shape[dim1_], grad_shape[dim2_]);
    std::swap(grad_strides[dim1_], grad_strides[dim2_]);
    TensorType transposed_type(grad_shape, grad_strides, grad_out.type().dtype());
    Tensor transposed_view(transposed_type, grad_out.storage(), false, grad_out.storage_offset());

    // Materialize as contiguous via strided TensorIterator copy
    auto dx_type = TensorType::contiguous(in_shape, grad_out.type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
    TensorIterator<const float> tv_it(transposed_view);
    TensorIterator<float> dx_it(dx);
    auto numel = dx.type().numel();
    for (int64_t i = 0; i < numel; ++i) {
        dx_it[i] = tv_it[i];
    }

    auto it = grads.find(input_ids_[0]);
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input_ids_[0]] = dx;
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
        RETURN_IF_ERROR(graph_[i]->apply(runtime, grads_));
    }

    for (size_t i = 0; i < graph_.size(); ++i) {
        const auto& node = graph_[i];
        const auto ids = node->input_ids();
        const auto& ins = node->inputs();
        for (size_t j = 0; j < ins.size(); ++j) {
            auto it = grads_.find(ids[j]);
            if (it != grads_.end()) {
                const_cast<Tensor&>(ins[j]).set_grad(it->second);
            }
        }
    }

    return {};
}

} // namespace axon
