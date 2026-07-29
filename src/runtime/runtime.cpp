#include "axon/runtime/runtime.h"
#include "axon/autograd/autograd.h"
#include "axon/nn/l1_loss.h"

namespace axon {

Tensor Runtime::empty(const std::vector<int64_t>& shape, DType dtype) {
    return Tensor::empty(*this, shape, dtype);
}

Tensor Runtime::zeros(const std::vector<int64_t>& shape, DType dtype) {
    return Tensor::zeros(*this, shape, dtype);
}

Tensor Runtime::ones(const std::vector<int64_t>& shape, DType dtype) {
    return Tensor::ones(*this, shape, dtype);
}

Tensor Runtime::randn(const std::vector<int64_t>& shape, DType dtype) {
    return Tensor::randn(*this, shape, dtype);
}

Expected<Tensor> Runtime::matmul(const Tensor& a, const Tensor& b) {
    return MatMulOp::forward(*this, a, b);
}

Expected<Tensor> Runtime::relu(const Tensor& x) {
    return ReLUOp::forward(*this, x);
}

Expected<Tensor> Runtime::gelu(const Tensor& x) {
    return GELUOp::forward(*this, x);
}

Expected<Tensor> Runtime::reshape(const Tensor& x, const std::vector<int64_t>& new_shape) {
    return ReshapeOp::forward(*this, x, new_shape);
}

Expected<Tensor> Runtime::transpose(const Tensor& x, int64_t dim1, int64_t dim2) {
    return TransposeOp::forward(*this, x, dim1, dim2);
}

Expected<Tensor> Runtime::mean(const Tensor& x, const std::vector<int64_t>& dims, bool keepdim) {
    return MeanOp::forward(*this, x, dims, keepdim);
}

Expected<Tensor> Runtime::l1_loss(const Tensor& pred, const Tensor& target) {
    return L1LossOp::forward(*this, pred, target);
}

Expected<Tensor> Runtime::add(const Tensor& a, const Tensor& b) {
    return AddOp::forward(*this, a, b);
}

Expected<Tensor> Runtime::conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                  int64_t stride, int64_t padding) {
    return Conv2DOp::forward(*this, input, weight, bias, stride, padding);
}

Expected<Tensor> Runtime::maxpool2d(const Tensor& input, int64_t kernel, int64_t stride) {
    return MaxPool2dOp::forward(*this, input, kernel, stride);
}

Expected<Tensor> Runtime::avgpool2d(const Tensor& input, int64_t kernel, int64_t stride) {
    return AvgPool2dOp::forward(*this, input, kernel, stride);
}

Expected<Tensor> Runtime::batchnorm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                                     const Tensor& running_mean, const Tensor& running_var,
                                     float momentum, float epsilon, bool training) {
    return BatchNormOp::forward(*this, input, gamma, beta, running_mean, running_var, momentum, epsilon, training);
}

Expected<Tensor> Runtime::layernorm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                                     float epsilon) {
    return LayerNormOp::forward(*this, input, gamma, beta, epsilon);
}

} // namespace axon
