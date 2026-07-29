#include "axon/nn/conv2d.h"
#include "axon/autograd/autograd.h"
#include "axon/runtime/runtime.h"

namespace axon {

Conv2D::Conv2D(Runtime& rt, size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, size_t padding, bool bias)
    : weight_(Tensor::randn(rt, {static_cast<int64_t>(out_channels), static_cast<int64_t>(in_channels),
                                  static_cast<int64_t>(kernel_size), static_cast<int64_t>(kernel_size)}), true)
    , bias_(Tensor::zeros(rt, {static_cast<int64_t>(out_channels)}), true)
    , stride_(stride)
    , padding_(padding)
    , has_bias_(bias)
{
    register_parameter("weight", &weight_);
    if (has_bias_) {
        register_parameter("bias", &bias_);
    }
}

Expected<Tensor> Conv2D::forward(Runtime& rt, const Tensor& x) {
    return Conv2DOp::forward(rt, x, weight_.tensor(), has_bias_ ? bias_.tensor() : Tensor(), stride_, padding_);
}

} // namespace axon
