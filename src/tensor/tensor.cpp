#include "axon/tensor/tensor.h"
#include "axon/runtime/runtime.h"
#include <cmath>
#include <cstring>
#include <random>

namespace axon {

TensorMetadata Tensor::default_type_ = TensorMetadata::contiguous({}, DType::Float32);

Tensor Tensor::empty(Runtime& rt, const std::vector<int64_t>& shape, DType dtype) {
    auto type = TensorMetadata::contiguous(shape, dtype);
    auto storage = rt.allocator().allocate(type);
    return Tensor(std::move(type), std::move(storage), false);
}

Tensor Tensor::zeros(Runtime& rt, const std::vector<int64_t>& shape, DType dtype) {
    auto t = empty(rt, shape, dtype);
    if (t.defined()) {
        std::memset(t.storage()->data, 0, t.storage()->size_bytes);
    }
    return t;
}

Tensor Tensor::ones(Runtime& rt, const std::vector<int64_t>& shape, DType dtype) {
    auto t = empty(rt, shape, dtype);
    if (t.defined() && dtype == DType::Float32) {
        auto* ptr = t.data<float>();
        for (int64_t i = 0; i < t.type().numel(); ++i) {
            ptr[i] = 1.0f;
        }
    }
    return t;
}

Tensor Tensor::randn(Runtime& rt, const std::vector<int64_t>& shape, DType dtype) {
    auto t = empty(rt, shape, dtype);
    if (t.defined() && dtype == DType::Float32) {
        static std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        auto* ptr = t.data<float>();
        for (int64_t i = 0; i < t.type().numel(); ++i) {
            ptr[i] = dist(gen);
        }
    }
    return t;
}

} // namespace axon
