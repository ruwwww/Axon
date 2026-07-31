#pragma once

#include <span>
#include <string>
#include <unordered_map>
#include "axon/backend/cpuid.h"
#include "axon/core/expected.h"
#include "axon/tensor/tensor.h"

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

    void register_kernel(const std::string& op_name, ISA isa, KernelFn fn);
    KernelFn lookup(const std::string& op_name, ISA isa) const;
    KernelFn dispatch(const std::string& op_name) const;

    void clear();

private:
    // Key: "op_name:isa"
    std::unordered_map<std::string, KernelFn> registry_;
};

} // namespace axon::cpu
