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
    Expected<Tensor> add(const Tensor& a, const Tensor& b);
    Expected<Tensor> conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                            int64_t stride = 1, int64_t padding = 0);
    Expected<Tensor> maxpool2d(const Tensor& input, int64_t kernel, int64_t stride);
    Expected<Tensor> avgpool2d(const Tensor& input, int64_t kernel, int64_t stride);
    Expected<Tensor> batchnorm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                               const Tensor& running_mean, const Tensor& running_var,
                               float momentum, float epsilon, bool training);
    Expected<Tensor> layernorm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                               float epsilon = 1e-5f);

private:
    Allocator allocator_;
    Autograd autograd_;
};

} // namespace axon
