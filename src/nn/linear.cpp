#include "axon/nn/linear.h"
#include "axon/autograd/autograd.h"
#include "axon/runtime/runtime.h"

namespace axon {

Linear::Linear(Runtime& rt, size_t in_features, size_t out_features, bool bias)
    : weight_(Tensor::randn(rt, {static_cast<int64_t>(in_features), static_cast<int64_t>(out_features)}), true)
    , bias_(Tensor::zeros(rt, {static_cast<int64_t>(out_features)}), true)
    , has_bias_(bias)
{
    register_parameter("weight", &weight_);
    if (has_bias_) {
        register_parameter("bias", &bias_);
    }
}

Expected<Tensor> Linear::forward(Runtime& rt, const Tensor& x) {
    auto y = rt.matmul(x, weight_.tensor());
    if (!y) return y.error();

    if (has_bias_) {
        y = AddOp::forward(rt, y.value(), bias_.tensor());
        if (!y) return y.error();
    }

    return y;
}

} // namespace axon
