#pragma once

#include <cstddef>
#include <utility>
#include "axon/tensor/tensor.h"

namespace axon {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> get(size_t index) = 0;
};

} // namespace axon
