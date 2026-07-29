#pragma once

#include <cstdint>
#include <vector>
#include "axon/core/expected.h"
#include "axon/core/types.h"
#include "axon/runtime/allocator.h"
#include "axon/tensor/tensor.h"

namespace axon {

class Runtime {
public:
    Allocator& allocator() { return allocator_; }

    Tensor empty(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor randn(const std::vector<int64_t>& shape, DType dtype = DType::Float32);

private:
    Allocator allocator_;
};

} // namespace axon
