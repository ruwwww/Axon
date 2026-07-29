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

} // namespace axon::cpu
