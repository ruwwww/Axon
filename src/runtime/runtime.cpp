#include "axon/runtime/runtime.h"

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

} // namespace axon
