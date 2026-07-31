#include "axon/nn/adamw.h"
#include <cstring>
#include <cmath>

namespace axon {

AdamW::AdamW(Runtime& rt, std::vector<Parameter*> params, float lr,
             float beta1, float beta2, float eps, float weight_decay)
    : rt_(rt), params_(std::move(params)), lr_(lr)
    , beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
{
    for (auto* p : params_) {
        const auto& shape = p->tensor().type().shape();
        auto m_type = TensorMetadata::contiguous(shape, p->tensor().type().dtype());
        auto v_type = TensorMetadata::contiguous(shape, p->tensor().type().dtype());
        Tensor m(m_type, rt_.allocator().allocate(m_type), false);
        Tensor v(v_type, rt_.allocator().allocate(v_type), false);
        memset(m.storage()->data, 0, m.storage()->size_bytes);
        memset(v.storage()->data, 0, v.storage()->size_bytes);
        state_.push_back({std::move(m), std::move(v)});
    }
}

Expected<void> AdamW::step() {
    ++t_;
    float bias_correction1 = 1.0f - std::pow(beta1_, t_);
    float bias_correction2 = 1.0f - std::pow(beta2_, t_);

    for (size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p->trainable()) continue;
        if (!p->has_grad()) continue;

        auto& w = p->tensor();
        auto& g = p->grad();
        auto& m = state_[i].m;
        auto& v = state_[i].v;

        auto n = w.type().numel();
        auto* w_ptr = w.data<float>();
        auto* g_ptr = g.data<float>();
        auto* m_ptr = m.data<float>();
        auto* v_ptr = v.data<float>();

        // Weight decay (decoupled)
        for (int64_t j = 0; j < n; ++j) {
            w_ptr[j] -= lr_ * weight_decay_ * w_ptr[j];
        }

        // Update moments
        for (int64_t j = 0; j < n; ++j) {
            m_ptr[j] = beta1_ * m_ptr[j] + (1.0f - beta1_) * g_ptr[j];
            v_ptr[j] = beta2_ * v_ptr[j] + (1.0f - beta2_) * g_ptr[j] * g_ptr[j];
        }

        // Apply bias-corrected update
        for (int64_t j = 0; j < n; ++j) {
            float m_hat = m_ptr[j] / bias_correction1;
            float v_hat = v_ptr[j] / bias_correction2;
            w_ptr[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }

    return {};
}

void AdamW::zero_grad() {
    for (auto* p : params_) {
        if (p->has_grad()) {
            memset(p->grad().storage()->data, 0, p->grad().storage()->size_bytes);
        }
    }
}

} // namespace axon
