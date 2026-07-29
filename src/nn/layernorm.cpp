#include "axon/nn/layernorm.h"
#include "axon/autograd/autograd.h"
#include "axon/runtime/runtime.h"

namespace axon {

LayerNorm::LayerNorm(Runtime& rt, const std::vector<int64_t>& normalized_shape, float epsilon)
    : gamma_(Tensor::ones(rt, normalized_shape), true)
    , beta_(Tensor::zeros(rt, normalized_shape), true)
    , normalized_size_(1)
    , epsilon_(epsilon)
{
    for (auto s : normalized_shape) normalized_size_ *= s;
    register_parameter("gamma", &gamma_);
    register_parameter("beta", &beta_);
}

Expected<Tensor> LayerNorm::forward(Runtime& rt, const Tensor& x) {
    return LayerNormOp::forward(rt, x, gamma_.tensor(), beta_.tensor(), epsilon_);
}

} // namespace axon
