#pragma once

#include <cstdint>
#include <vector>
#include "axon/tensor/tensor.h"

namespace axon {

template <typename T>
class TensorIterator {
public:
    TensorIterator() = default;

    explicit TensorIterator(const Tensor& tensor)
        : data_(static_cast<T*>(tensor.storage()->data) + tensor.storage_offset())
        , numel_(tensor.type().numel())
        , shape_(tensor.type().shape())
        , strides_(tensor.type().strides())
        , contiguous_(check_contiguous()) {}

    bool is_contiguous() const { return contiguous_; }
    int64_t numel() const { return numel_; }
    size_t ndim() const { return shape_.size(); }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }

    T& operator[](int64_t index) {
        if (contiguous_) {
            return data_[index];
        }
        return data_[offset_for(index)];
    }

    const T& operator[](int64_t index) const {
        if (contiguous_) {
            return data_[index];
        }
        return data_[offset_for(index)];
    }

private:
    int64_t offset_for(int64_t index) const {
        int64_t remaining = index;
        int64_t offset = 0;
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            int64_t idx = remaining % shape_[d];
            remaining /= shape_[d];
            offset += idx * strides_[d];
        }
        return offset;
    }

    bool check_contiguous() const {
        if (shape_.empty()) return true;
        int64_t expected = 1;
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            if (strides_[d] != expected) return false;
            expected *= shape_[d];
        }
        return true;
    }

    T* data_ = nullptr;
    int64_t numel_ = 0;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    bool contiguous_ = true;
};

} // namespace axon
