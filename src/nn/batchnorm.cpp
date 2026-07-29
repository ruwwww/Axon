#include "axon/nn/batchnorm.h"
#include "axon/autograd/autograd.h"
#include "axon/runtime/runtime.h"
#include <cstring>

namespace axon {

BatchNorm::BatchNorm(Runtime& rt, size_t channels, float momentum, float epsilon)
    : gamma_(Tensor::ones(rt, {static_cast<int64_t>(channels)}), true)
    , beta_(Tensor::zeros(rt, {static_cast<int64_t>(channels)}), true)
    , running_mean_(Tensor::zeros(rt, {static_cast<int64_t>(channels)}))
    , running_var_(Tensor::ones(rt, {static_cast<int64_t>(channels)}))
    , momentum_(momentum)
    , epsilon_(epsilon)
{
    register_parameter("gamma", &gamma_);
    register_parameter("beta", &beta_);
}

Expected<Tensor> BatchNorm::forward(Runtime& rt, const Tensor& x) {
    return BatchNormOp::forward(rt, x, gamma_.tensor(), beta_.tensor(),
                                running_mean_, running_var_, momentum_, epsilon_, is_training());
}

} // namespace axon
