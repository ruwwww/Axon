#pragma once

#include <cstdint>
#include <vector>
#include <cassert>
#include "axon/core/types.h"

namespace axon {

class TensorType {
public:
    TensorType() = default;

    TensorType(
        std::vector<int64_t> shape,
        std::vector<int64_t> strides,
        DType dtype,
        Device device = Device::CPU,
        QuantFormat quant = QuantFormat::None
    ) : shape_(std::move(shape))
      , strides_(std::move(strides))
      , dtype_(dtype)
      , device_(device)
      , quant_(quant) {}

    static TensorType contiguous(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU) {
        auto strides = compute_strides(shape);
        return TensorType(std::move(shape), std::move(strides), dtype, device, QuantFormat::None);
    }

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    QuantFormat quant() const { return quant_; }

    size_t ndim() const { return shape_.size(); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape_) n *= s;
        return n;
    }

    size_t size_bytes() const {
        if (quant_ != QuantFormat::None) {
            size_t numel_val = static_cast<size_t>(numel());
            size_t num_blocks_32 = (numel_val + 31) / 32;
            size_t num_blocks_256 = (numel_val + 255) / 256;
            switch (quant_) {
                case QuantFormat::Q8_0: return num_blocks_32 * 34;
                case QuantFormat::Q4_0: return num_blocks_32 * 18;
                case QuantFormat::Q2_K: return num_blocks_256 * 84;
                case QuantFormat::Q3_K: return num_blocks_256 * 110;
                case QuantFormat::Q4_K: return num_blocks_256 * 144;
                case QuantFormat::Q5_K: return num_blocks_256 * 176;
                case QuantFormat::Q6_K: return num_blocks_256 * 210;
                default: return 0;
            }
        }
        return static_cast<size_t>(numel()) * size_of(dtype_);
    }

private:
    static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::CPU;
    QuantFormat quant_ = QuantFormat::None;
};

} // namespace axon
