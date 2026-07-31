#pragma once

#include <span>
#include <string>
#include <unordered_map>
#include "axon/backend/cpuid.h"
#include "axon/core/expected.h"
#include "axon/core/types.h"
#include "axon/tensor/tensor.h"

namespace axon::cpu {

enum class OpId : uint16_t {
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
    ReLU,
    GELU,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    BatchNorm,
    LayerNorm,
    Dropout,
    Flatten,
    Embedding,
    Reshape,
    Transpose,
    Mean,
    Sum,
    Max,
    CrossEntropyLoss,
    MSELoss,
    L1Loss,
    MatMulQ4_0,
    MatMulQ4_K,
    MatMulQ5_K,
    MatMulBLAS
};

enum class Provider : uint8_t {
    AxonNative = 0,
    BLAS = 1,
    CUDA = 2,
    OpenCL = 3
};

struct KernelKey {
    OpId op = OpId::MatMul;
    Device device = Device::CPU;
    DType dtype = DType::Float32;
    Provider provider = Provider::AxonNative;

    bool operator==(const KernelKey& o) const = default;
};

} // namespace axon::cpu

namespace std {

template <>
struct hash<axon::cpu::KernelKey> {
    size_t operator()(const axon::cpu::KernelKey& k) const noexcept {
        uint64_t val = (static_cast<uint64_t>(k.op)) |
                       (static_cast<uint64_t>(k.device) << 16) |
                       (static_cast<uint64_t>(k.dtype) << 24) |
                       (static_cast<uint64_t>(k.provider) << 32);
        return std::hash<uint64_t>{}(val);
    }
};

} // namespace std

namespace axon::cpu {

struct KernelContext {
    std::span<Tensor> outputs;
    std::span<const Tensor> inputs;
    void* attributes = nullptr;
};

using KernelFn = Expected<void>(*)(KernelContext& ctx);

class KernelRegistry {
public:
    static KernelRegistry& instance();

    void register_kernel(KernelKey key, KernelFn fn);
    KernelFn lookup(KernelKey key) const;
    KernelFn dispatch(KernelKey key) const;

    void register_kernel(const std::string& op_name, ISA isa, KernelFn fn);
    KernelFn lookup(const std::string& op_name, ISA isa) const;
    KernelFn dispatch(const std::string& op_name) const;

    void clear();

private:
    std::unordered_map<KernelKey, KernelFn> registry_;
};

void register_cpu_kernels();

} // namespace axon::cpu
