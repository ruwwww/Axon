#pragma once

#include <cstdint>
#include <vector>
#include "axon/autograd/autograd.h"
#include "axon/core/expected.h"
#include "axon/core/types.h"
#include "axon/runtime/allocator.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime {
public:
    Allocator& allocator() { return allocator_; }
    Autograd& autograd() { return autograd_; }

    Tensor empty(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor randn(const std::vector<int64_t>& shape, DType dtype = DType::Float32);

    Expected<Tensor> matmul(const Tensor& a, const Tensor& b);
    Expected<Tensor> relu(const Tensor& x);

private:
    Allocator allocator_;
    Autograd autograd_;
};

} // namespace axon
