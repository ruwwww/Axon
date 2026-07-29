#include "axon/nn/sgd.h"
#include <cstring>

namespace axon {

SGD::SGD(Runtime& rt, std::vector<Parameter*> params, float lr, float momentum)
    : rt_(rt), params_(std::move(params)), lr_(lr), momentum_(momentum)
{
    if (momentum_ > 0.0f) {
        for (auto* p : params_) {
            auto type = TensorType::contiguous(p->tensor().type().shape(), p->tensor().type().dtype());
            auto buf = Tensor(type, rt_.allocator().allocate(type), false);
            memset(buf.storage()->data, 0, buf.storage()->size_bytes);
            momentum_bufs_.push_back(std::move(buf));
        }
    }
}

Expected<void> SGD::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p->trainable()) continue;
        if (!p->has_grad()) continue;

        auto& w = p->tensor();
        auto& g = p->grad();
        auto n = w.type().numel();
        auto* w_ptr = w.data<float>();
        auto* g_ptr = g.data<float>();

        if (momentum_ > 0.0f) {
            auto& buf = momentum_bufs_[i];
            auto* buf_ptr = buf.data<float>();
            for (int64_t j = 0; j < n; ++j) {
                buf_ptr[j] = momentum_ * buf_ptr[j] + lr_ * g_ptr[j];
                w_ptr[j] -= buf_ptr[j];
            }
        } else {
            for (int64_t j = 0; j < n; ++j) {
                w_ptr[j] -= lr_ * g_ptr[j];
            }
        }
    }
    return {};
}

void SGD::zero_grad() {
    for (auto* p : params_) {
        if (p->has_grad()) {
            memset(p->grad().storage()->data, 0, p->grad().storage()->size_bytes);
        }
    }
}

} // namespace axon
