#pragma once

#include <cstddef>
#include <cstdint>
#include "axon/core/expected.h"
#include "axon/core/types.h"
#include "axon/tensor/tensor.h"

namespace axon::cpu {

Expected<void> add(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> sub(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> mul(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> div(Tensor& out, const Tensor& a, const Tensor& b);

Expected<void> matmul(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> relu(Tensor& out, const Tensor& x);
Expected<void> gelu(Tensor& out, const Tensor& x);

Expected<void> log_softmax(Tensor& out, const Tensor& x);
Expected<void> softmax(Tensor& out, const Tensor& x);

// Convolution
Expected<void> conv2d(Tensor& out, const Tensor& input, const Tensor& weight,
                      int64_t stride, int64_t padding);

// Pooling
Expected<void> maxpool2d(Tensor& out, const Tensor& input,
                         int64_t kernel, int64_t stride);
Expected<void> avgpool2d(Tensor& out, const Tensor& input,
                         int64_t kernel, int64_t stride);

// Quantization
size_t quantized_size(size_t num_elements, QuantFormat format);
size_t quantized_size_2d(int64_t M, int64_t K, QuantFormat format);
Expected<void> quantize(Tensor& dst, const Tensor& src, QuantFormat format);
Expected<void> dequantize(Tensor& dst, const Tensor& src);
Expected<void> matmul_q4(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> matmul_q2_K(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> matmul_q3_K(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> matmul_q4_K(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> matmul_q5_K(Tensor& out, const Tensor& a, const Tensor& b);
Expected<void> matmul_q6_K(Tensor& out, const Tensor& a, const Tensor& b);

// Reductions
Expected<void> reduce_mean(Tensor& out, const Tensor& input, const std::vector<int64_t>& dims);

// Normalization
Expected<void> batchnorm(Tensor& out, const Tensor& input,
                         const Tensor& gamma, const Tensor& beta,
                         const Tensor& running_mean, const Tensor& running_var,
                         float momentum, float epsilon, bool training);

Expected<void> layernorm(Tensor& out, const Tensor& input,
                         const Tensor& gamma, const Tensor& beta,
                         float epsilon);

} // namespace axon::cpu
