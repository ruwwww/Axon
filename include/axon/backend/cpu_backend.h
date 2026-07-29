#pragma once

#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

namespace axon::cpu {

Expected<void> add(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> sub(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> mul(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> div(Tensor& out, const Tensor& a, const Tensor& b);

Expected<void> matmul(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> relu(Tensor& out, const Tensor& x);

Expected<void> log_softmax(Tensor& out, const Tensor& x);
Expected<void> softmax(Tensor& out, const Tensor& x);

// Convolution
Expected<void> conv2d(Tensor& out, const Tensor& input, const Tensor& weight,
                      int64_t stride, int64_t padding);

// Pooling
Expected<void> maxpool2d(Tensor& out, const Tensor& input,
                         int64_t kernel, int64_t stride);

// Normalization
Expected<void> batchnorm(Tensor& out, const Tensor& input,
                         const Tensor& gamma, const Tensor& beta,
                         const Tensor& running_mean, const Tensor& running_var,
                         float momentum, float epsilon, bool training);

Expected<void> layernorm(Tensor& out, const Tensor& input,
                         const Tensor& gamma, const Tensor& beta,
                         float epsilon);

} // namespace axon::cpu
