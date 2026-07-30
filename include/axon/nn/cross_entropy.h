#pragma once

#include "axon/autograd/autograd.h"
#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime;

struct CrossEntropyLossOp {
    static Expected<Tensor> forward(Runtime& rt, const Tensor& logits, const Tensor& targets);
};

} // namespace axon
