#pragma once

#include <cstdint>

namespace axon {

using TensorId = uint64_t;

enum class DType : uint8_t {
    Float32,
    Float64,
    Int32,
    Int64,
    Int16,
    Int8,
    UInt8,
};

inline size_t size_of(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::Int16:   return 2;
        case DType::Int8:    return 1;
        case DType::UInt8:   return 1;
    }
    return 0;
}

enum class Device : uint8_t {
    CPU,
};

enum class QuantFormat : uint8_t {
    None,
    Q8_0,
    Q6_K,
    Q5_K,
    Q4_0,
    Q4_K,
    Q3_K,
    Q2_K,
};

} // namespace axon
