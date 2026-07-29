#include "axon/runtime/runtime.h"
#include "axon/autograd/autograd.h"

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

} // namespace axon
