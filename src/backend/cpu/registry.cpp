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

void register_scalar_elementwise_kernels();
void register_avx2_elementwise_kernels();
void register_scalar_gemm_kernels();
void register_avx2_gemm_kernels();
void register_avx2_quants_kernels();

static bool g_kernels_registered = false;

void register_cpu_kernels() {
    if (!g_kernels_registered) {
        g_kernels_registered = true;
        register_scalar_elementwise_kernels();
        register_avx2_elementwise_kernels();
        register_scalar_gemm_kernels();
        register_avx2_gemm_kernels();
        register_avx2_quants_kernels();
    }
}

KernelFn KernelRegistry::lookup(const std::string& op_name, ISA isa) const {
    register_cpu_kernels();
    auto it = registry_.find(make_key(op_name, isa));
    if (it != registry_.end()) {
        return it->second;
    }
    return nullptr;
}

KernelFn KernelRegistry::dispatch(const std::string& op_name) const {
    register_cpu_kernels();
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
    g_kernels_registered = false;
}

} // namespace axon::cpu
