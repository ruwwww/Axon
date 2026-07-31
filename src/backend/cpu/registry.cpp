#include "axon/backend/registry.h"

namespace axon::cpu {

static std::string make_key(const std::string& op_name, ISA isa) {
    return op_name + ":" + std::to_string(static_cast<int>(isa));
}

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry reg;
    return reg;
}

void KernelRegistry::register_kernel(const std::string& op_name, ISA isa, KernelFn fn) {
    registry_[make_key(op_name, isa)] = fn;
}

KernelFn KernelRegistry::lookup(const std::string& op_name, ISA isa) const {
    auto it = registry_.find(make_key(op_name, isa));
    if (it != registry_.end()) {
        return it->second;
    }
    return nullptr;
}

KernelFn KernelRegistry::dispatch(const std::string& op_name) const {
    ISA best = get_best_isa();

    // Try best ISA first
    if (best == ISA::AVX2) {
        KernelFn fn = lookup(op_name, ISA::AVX2);
        if (fn) return fn;
    }

    // Fall back to Scalar
    return lookup(op_name, ISA::Scalar);
}

void KernelRegistry::clear() {
    registry_.clear();
}

} // namespace axon::cpu
