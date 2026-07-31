#include "axon/nn/dropout.h"
#include "axon/runtime/runtime.h"
#include <random>

namespace axon {

Dropout::Dropout(float p) : p_(p) {}

Expected<Tensor> Dropout::forward(Runtime& rt, const Tensor& x) {
    if (!is_training() || p_ <= 0.0f) {
        // No-op during eval or when p=0
        auto out_type = TensorMetadata::contiguous(x.type().shape(), x.type().dtype());
        auto out = Tensor(out_type, rt.allocator().allocate(out_type), x.requires_grad());
        auto* x_ptr = x.data<const float>();
        auto* o_ptr = out.data<float>();
        auto n = x.type().numel();
        for (int64_t i = 0; i < n; ++i) o_ptr[i] = x_ptr[i];
        return out;
    }

    auto out_type = TensorMetadata::contiguous(x.type().shape(), x.type().dtype());
    auto out = Tensor(out_type, rt.allocator().allocate(out_type), x.requires_grad());
    auto* x_ptr = x.data<const float>();
    auto* o_ptr = out.data<float>();
    auto n = x.type().numel();

    float scale = 1.0f / (1.0f - p_);
    static thread_local std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0f - p_);

    for (int64_t i = 0; i < n; ++i) {
        o_ptr[i] = dist(gen) ? x_ptr[i] * scale : 0.0f;
    }

    return out;
}

} // namespace axon
