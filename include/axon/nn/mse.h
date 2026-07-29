#pragma once

#include "axon/autograd/autograd.h"
#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime;

struct MSELossOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& pred, const Tensor& target);
    static Expected<void> backward(Runtime& rt, const GraphNode& node, GradientMap& grads);
};

} // namespace axon
