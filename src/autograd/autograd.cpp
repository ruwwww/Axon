#include "axon/autograd/autograd.h"
#include "axon/backend/cpu_backend.h"
#include "axon/nn/cross_entropy.h"
#include "axon/nn/mse.h"
#include "axon/runtime/runtime.h"
#include <cmath>
#include <limits>

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

    // Add bias
    if (bias.defined() && bias.type().numel() > 0) {
        auto* o_ptr = out.data<float>();
        auto* b_ptr = bias.data<const float>();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc = 0; oc < OC; ++oc) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        o_ptr[n * OC * OH * OW + oc * OH * OW + oh * OW + ow] += b_ptr[oc];
                    }
                }
            }
        }
    }

    if (need_grad) {
        GraphNode node;
        node.op = OpType::Conv2D;
        node.inputs = {input, weight, bias};
        node.output = out;
        node.runtime = &rt;
        // Store stride/padding in op_data as a 2-element tensor
        auto meta_type = TensorType::contiguous({2}, DType::Int64);
        Tensor meta(meta_type, rt.allocator().allocate(meta_type), false);
        meta.data<int64_t>()[0] = stride;
        meta.data<int64_t>()[1] = padding;
        node.op_data = meta;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> Conv2DOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& input = node.inputs[0];
    const Tensor& weight = node.inputs[1];
    const Tensor& bias = node.inputs[2];
    int64_t stride = node.op_data.data<const int64_t>()[0];
    int64_t padding = node.op_data.data<const int64_t>()[1];

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& in_shape = input.type().shape();
    const auto& w_shape = weight.type().shape();
    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OC = w_shape[0], IC = w_shape[1], KH = w_shape[2], KW = w_shape[3];
    auto OH = grad_out.type().shape()[2], OW = grad_out.type().shape()[3];

    // d_input: conv transpose of grad_out with weight
    if (input.requires_grad()) {
        auto di_type = TensorType::contiguous({N, C, H, W}, input.type().dtype());
        Tensor di(di_type, rt.allocator().allocate(di_type), false);
        auto* di_ptr = di.data<float>();
        auto* go_ptr = grad_out.data<const float>();
        auto* w_ptr = weight.data<const float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        float sum = 0.0f;
                        for (int64_t oc = 0; oc < OC; ++oc) {
                            for (int64_t kh = 0; kh < KH; ++kh) {
                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    int64_t oh = (h + padding - kh);
                                    int64_t ow = (w + padding - kw);
                                    if (oh % stride == 0 && ow % stride == 0) {
                                        oh /= stride;
                                        ow /= stride;
                                        if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
                                            sum += go_ptr[n * OC * OH * OW + oc * OH * OW + oh * OW + ow]
                                                 * w_ptr[oc * IC * KH * KW + c * KH * KW + kh * KW + kw];
                                        }
                                    }
                                }
                            }
                        }
                        di_ptr[n * C * H * W + c * H * W + h * W + w] = sum;
                    }
                }
            }
        }

        auto it = grads.find(input.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, di);
        } else {
            grads[input.id()] = di;
        }
    }

    // d_weight
    if (weight.requires_grad()) {
        auto dw_type = TensorType::contiguous({OC, IC, KH, KW}, weight.type().dtype());
        Tensor dw(dw_type, rt.allocator().allocate(dw_type), false);
        auto* dw_ptr = dw.data<float>();
        auto* inp_ptr = input.data<const float>();
        auto* go_ptr = grad_out.data<const float>();

        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t ic = 0; ic < IC; ++ic) {
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        float sum = 0.0f;
                        for (int64_t n = 0; n < N; ++n) {
                            for (int64_t oh = 0; oh < OH; ++oh) {
                                for (int64_t ow = 0; ow < OW; ++ow) {
                                    int64_t ih = oh * stride + kh - padding;
                                    int64_t iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        sum += inp_ptr[n * C * H * W + ic * H * W + ih * W + iw]
                                             * go_ptr[n * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                                    }
                                }
                            }
                        }
                        dw_ptr[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw] = sum;
                    }
                }
            }
        }

        auto it = grads.find(weight.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dw);
        } else {
            grads[weight.id()] = dw;
        }
    }

    // d_bias: sum grad_out over N, H, W per output channel
    if (bias.defined() && bias.requires_grad() && bias.type().numel() > 0) {
        auto db_type = TensorType::contiguous({OC}, DType::Float32);
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        auto* db_ptr = db.data<float>();
        auto* go_ptr = grad_out.data<const float>();

        for (int64_t oc = 0; oc < OC; ++oc) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        sum += go_ptr[n * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                    }
                }
            }
            db_ptr[oc] = sum;
        }

        auto it = grads.find(bias.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[bias.id()] = db;
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
        GraphNode node;
        node.op = OpType::MaxPool2d;
        node.inputs = {input};
        node.output = out;
        node.runtime = &rt;
        // Store kernel/stride in op_data
        auto meta_type = TensorType::contiguous({2}, DType::Int64);
        Tensor meta(meta_type, rt.allocator().allocate(meta_type), false);
        meta.data<int64_t>()[0] = kernel;
        meta.data<int64_t>()[1] = stride;
        node.op_data = meta;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> MaxPool2dOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& input = node.inputs[0];
    int64_t kernel = node.op_data.data<const int64_t>()[0];
    int64_t stride = node.op_data.data<const int64_t>()[1];

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) return {};

    const auto& in_shape = input.type().shape();
    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OH = node.output.type().shape()[2], OW = node.output.type().shape()[3];

    auto dx_type = TensorType::contiguous({N, C, H, W}, input.type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
    auto* dx_ptr = dx.data<float>();
    Tensor grad_out = grad_it->second;
    auto* inp_ptr = input.data<const float>();
    auto* go_ptr = grad_out.data<const float>();

    // Upstream gradient only flows to the max element in each pool window
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    // Find argmax in this window
                    int64_t argmax_h = 0, argmax_w = 0;
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t kh = 0; kh < kernel; ++kh) {
                        for (int64_t kw = 0; kw < kernel; ++kw) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            if (ih < H && iw < W) {
                                float val = inp_ptr[n * C * H * W + c * H * W + ih * W + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    argmax_h = kh;
                                    argmax_w = kw;
                                }
                            }
                        }
                    }
                    int64_t ih = oh * stride + argmax_h;
                    int64_t iw = ow * stride + argmax_w;
                    dx_ptr[n * C * H * W + c * H * W + ih * W + iw] += go_ptr[n * C * OH * OW + c * OH * OW + oh * OW + ow];
                }
            }
        }
    }

    auto it = grads.find(input.id());
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input.id()] = dx;
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
        GraphNode node;
        node.op = OpType::AvgPool2d;
        node.inputs = {input};
        node.output = out;
        node.runtime = &rt;
        auto meta_type = TensorType::contiguous({2}, DType::Int64);
        Tensor meta(meta_type, rt.allocator().allocate(meta_type), false);
        meta.data<int64_t>()[0] = kernel;
        meta.data<int64_t>()[1] = stride;
        node.op_data = meta;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> AvgPool2dOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& input = node.inputs[0];
    int64_t kernel = node.op_data.data<const int64_t>()[0];
    int64_t stride = node.op_data.data<const int64_t>()[1];

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) return {};

    const auto& in_shape = input.type().shape();
    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OH = node.output.type().shape()[2], OW = node.output.type().shape()[3];

    auto dx_type = TensorType::contiguous({N, C, H, W}, input.type().dtype());
    Tensor dx(dx_type, rt.allocator().allocate(dx_type), false);
    auto* dx_ptr = dx.data<float>();
    Tensor grad_out = grad_it->second;
    auto* go_ptr = grad_out.data<const float>();

    float inv_k = 1.0f / static_cast<float>(kernel * kernel);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float g = go_ptr[n * C * OH * OW + c * OH * OW + oh * OW + ow];
                    for (int64_t kh = 0; kh < kernel; ++kh) {
                        for (int64_t kw = 0; kw < kernel; ++kw) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            if (ih < H && iw < W) {
                                dx_ptr[n * C * H * W + c * H * W + ih * W + iw] += g * inv_k;
                            }
                        }
                    }
                }
            }
        }
    }

    auto it = grads.find(input.id());
    if (it != grads.end()) {
        cpu::add(it->second, it->second, dx);
    } else {
        grads[input.id()] = dx;
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
        GraphNode node;
        node.op = OpType::BatchNorm;
        node.inputs = {input, gamma, beta, running_mean, running_var};
        node.output = out;
        node.runtime = &rt;
        // Store momentum, epsilon, training flag in op_data
        auto meta_type = TensorType::contiguous({3}, DType::Float32);
        Tensor meta(meta_type, rt.allocator().allocate(meta_type), false);
        meta.data<float>()[0] = momentum;
        meta.data<float>()[1] = epsilon;
        meta.data<float>()[2] = training ? 1.0f : 0.0f;
        node.op_data = meta;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> BatchNormOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& input = node.inputs[0];
    const Tensor& gamma = node.inputs[1];
    const Tensor& beta = node.inputs[2];
    float epsilon = node.op_data.data<const float>()[1];

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& in_shape = input.type().shape();
    int64_t N = in_shape[0], C = in_shape[1];
    int64_t spatial = 1;
    for (size_t i = 2; i < in_shape.size(); ++i) spatial *= in_shape[i];
    int64_t num_elements = N * spatial;

    auto* inp = input.data<const float>();
    auto* g = gamma.data<const float>();
    auto* go = grad_out.data<const float>();

    // Compute mean and variance
    std::vector<float> mean(C, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t s = 0; s < spatial; ++s)
                sum += inp[n * C * spatial + c * spatial + s];
            mean[c] += sum;
        }
    for (int64_t c = 0; c < C; ++c) mean[c] /= static_cast<float>(num_elements);

    std::vector<float> var(C, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t s = 0; s < spatial; ++s) {
                float diff = inp[n * C * spatial + c * spatial + s] - mean[c];
                sum += diff * diff;
            }
            var[c] += sum;
        }
    for (int64_t c = 0; c < C; ++c) var[c] /= static_cast<float>(num_elements);

    std::vector<float> inv_std(C);
    for (int64_t c = 0; c < C; ++c)
        inv_std[c] = 1.0f / std::sqrt(var[c] + epsilon);

    // d_gamma = sum(grad_out * x_hat) over N and spatial
    if (gamma.requires_grad()) {
        auto dg_type = TensorType::contiguous({C}, DType::Float32);
        Tensor dg(dg_type, rt.allocator().allocate(dg_type), false);
        auto* dg_ptr = dg.data<float>();

        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t s = 0; s < spatial; ++s) {
                    float x_hat = (inp[n * C * spatial + c * spatial + s] - mean[c]) * inv_std[c];
                    sum += go[n * C * spatial + c * spatial + s] * x_hat;
                }
            dg_ptr[c] = sum;
        }

        auto it = grads.find(gamma.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dg);
        } else {
            grads[gamma.id()] = dg;
        }
    }

    // d_beta = sum(grad_out) over N and spatial
    if (beta.requires_grad()) {
        auto db_type = TensorType::contiguous({C}, DType::Float32);
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        auto* db_ptr = db.data<float>();

        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n)
                for (int64_t s = 0; s < spatial; ++s)
                    sum += go[n * C * spatial + c * spatial + s];
            db_ptr[c] = sum;
        }

        auto it = grads.find(beta.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[beta.id()] = db;
        }
    }

    // d_input
    if (input.requires_grad()) {
        auto di_type = TensorType::contiguous(in_shape, DType::Float32);
        Tensor di(di_type, rt.allocator().allocate(di_type), false);
        auto* di_ptr = di.data<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float inv_N = 1.0f / static_cast<float>(num_elements);
                float g_val = g[c];
                float istd = inv_std[c];
                float x_mean_sum = 0.0f;
                float go_sum = 0.0f;

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    float x_hat = (inp[idx] - mean[c]) * istd;
                    x_mean_sum += x_hat;
                    go_sum += go[idx];
                }

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    float x_hat = (inp[idx] - mean[c]) * istd;
                    di_ptr[idx] = g_val * istd * inv_N * (
                        static_cast<float>(num_elements) * go[idx] - go_sum - x_hat * x_mean_sum
                    );
                }
            }
        }

        auto it = grads.find(input.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, di);
        } else {
            grads[input.id()] = di;
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
        GraphNode node;
        node.op = OpType::LayerNorm;
        node.inputs = {input, gamma, beta};
        node.output = out;
        node.runtime = &rt;
        // Store epsilon in op_data
        auto meta_type = TensorType::contiguous({1}, DType::Float32);
        Tensor meta(meta_type, rt.allocator().allocate(meta_type), false);
        meta.data<float>()[0] = epsilon;
        node.op_data = meta;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> LayerNormOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
    const Tensor& input = node.inputs[0];
    const Tensor& gamma = node.inputs[1];
    const Tensor& beta = node.inputs[2];
    float epsilon = node.op_data.data<const float>()[0];

    auto grad_it = grads.find(node.output.id());
    if (grad_it == grads.end()) return {};
    Tensor grad_out = grad_it->second;

    const auto& in_shape = input.type().shape();
    auto N = in_shape[0];
    int64_t D = 1;
    for (size_t i = 1; i < in_shape.size(); ++i) D *= in_shape[i];

    auto* inp = input.data<const float>();
    auto* g = gamma.data<const float>();
    auto* go = grad_out.data<const float>();

    // Compute mean and variance per sample
    std::vector<float> mean(N), var(N), inv_std(N);
    for (int64_t n = 0; n < N; ++n) {
        float m = 0.0f;
        for (int64_t d = 0; d < D; ++d) m += inp[n * D + d];
        mean[n] = m / static_cast<float>(D);

        float v = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float diff = inp[n * D + d] - mean[n];
            v += diff * diff;
        }
        var[n] = v / static_cast<float>(D);
        inv_std[n] = 1.0f / std::sqrt(var[n] + epsilon);
    }

    // d_gamma = sum(grad_out * x_hat) over N
    if (gamma.requires_grad()) {
        auto dg_type = TensorType::contiguous({D}, DType::Float32);
        Tensor dg(dg_type, rt.allocator().allocate(dg_type), false);
        auto* dg_ptr = dg.data<float>();

        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                float x_hat = (inp[n * D + d] - mean[n]) * inv_std[n];
                sum += go[n * D + d] * x_hat;
            }
            dg_ptr[d] = sum;
        }

        auto it = grads.find(gamma.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, dg);
        } else {
            grads[gamma.id()] = dg;
        }
    }

    // d_beta = sum(grad_out) over N
    if (beta.requires_grad()) {
        auto db_type = TensorType::contiguous({D}, DType::Float32);
        Tensor db(db_type, rt.allocator().allocate(db_type), false);
        auto* db_ptr = db.data<float>();

        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) sum += go[n * D + d];
            db_ptr[d] = sum;
        }

        auto it = grads.find(beta.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, db);
        } else {
            grads[beta.id()] = db;
        }
    }

    // d_input
    if (input.requires_grad()) {
        auto di_type = TensorType::contiguous(in_shape, DType::Float32);
        Tensor di(di_type, rt.allocator().allocate(di_type), false);
        auto* di_ptr = di.data<float>();

        for (int64_t n = 0; n < N; ++n) {
            float inv_D = 1.0f / static_cast<float>(D);
            float istd = inv_std[n];

            // Precompute sums
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                float x_hat = (inp[n * D + d] - mean[n]) * istd;
                sum1 += go[n * D + d] * g[d];
                sum2 += go[n * D + d] * g[d] * x_hat;
            }

            for (int64_t d = 0; d < D; ++d) {
                float x_hat = (inp[n * D + d] - mean[n]) * istd;
                di_ptr[n * D + d] = istd * (
                    go[n * D + d] * g[d] - inv_D * sum1 - x_hat * inv_D * sum2
                );
            }
        }

        auto it = grads.find(input.id());
        if (it != grads.end()) {
            cpu::add(it->second, it->second, di);
        } else {
            grads[input.id()] = di;
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
        GraphNode node;
        node.op = OpType::GELU;
        node.inputs = {x};
        node.output = out;
        node.runtime = &rt;
        rt.autograd().graph().append(std::move(node));
    }

    return out;
}

Expected<void> GELUOp::backward(Runtime& rt, const GraphNode& node, GradientMap& grads) {
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
    constexpr float alpha = 0.79788456f;
    constexpr float beta = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        float xi = x_ptr[i];
        float x3 = xi * xi * xi;
        float g = alpha * (xi + beta * x3);
        float t = std::tanh(g);
        float t2 = 1.0f - t * t;
        float gprime = alpha * (1.0f + 3.0f * beta * xi * xi);
        float df = 0.5f * (1.0f + t + xi * t2 * gprime);
        dx_ptr[i] = go_ptr[i] * df;
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
            case OpType::Conv2D: {
                RETURN_IF_ERROR(Conv2DOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::MaxPool2d: {
                RETURN_IF_ERROR(MaxPool2dOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::AvgPool2d: {
                RETURN_IF_ERROR(AvgPool2dOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::BatchNorm: {
                RETURN_IF_ERROR(BatchNormOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::LayerNorm: {
                RETURN_IF_ERROR(LayerNormOp::backward(*node.runtime, node, grads_));
                break;
            }
            case OpType::GELU: {
                RETURN_IF_ERROR(GELUOp::backward(*node.runtime, node, grads_));
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
