#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime;

using GradientMap = std::unordered_map<TensorId, Tensor>;

class Node {
public:
    virtual ~Node() = default;
    virtual Expected<void> apply(Runtime& runtime, GradientMap& grads) = 0;
    virtual std::vector<Tensor>& inputs() = 0;
    virtual const std::vector<Tensor>& inputs() const = 0;
    virtual std::vector<TensorId> input_ids() const = 0;
    virtual std::string name() const { return "Node"; }
};

} // namespace axon
