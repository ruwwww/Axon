#include "axon/backend/registry.h"

namespace axon::cpu {

static KernelKey string_to_kernel_key(const std::string& op_name) {
    KernelKey key;
    key.device = Device::CPU;
    key.dtype = DType::Float32;
    key.provider = Provider::AxonNative;

    if (op_name == "add") key.op = OpId::Add;
    else if (op_name == "mul") key.op = OpId::Mul;
    else if (op_name == "sub") key.op = OpId::Sub;
    else if (op_name == "div") key.op = OpId::Div;
    else if (op_name == "matmul") key.op = OpId::MatMul;
    else if (op_name == "relu") key.op = OpId::ReLU;
    else if (op_name == "gelu") key.op = OpId::GELU;
    else if (op_name == "conv2d") key.op = OpId::Conv2D;
    else if (op_name == "maxpool2d") key.op = OpId::MaxPool2D;
    else if (op_name == "avgpool2d") key.op = OpId::AvgPool2D;
    else if (op_name == "batchnorm") key.op = OpId::BatchNorm;
    else if (op_name == "layernorm") key.op = OpId::LayerNorm;
    else if (op_name == "dropout") key.op = OpId::Dropout;
    else if (op_name == "flatten") key.op = OpId::Flatten;
    else if (op_name == "embedding") key.op = OpId::Embedding;
    else if (op_name == "reshape") key.op = OpId::Reshape;
    else if (op_name == "transpose") key.op = OpId::Transpose;
    else if (op_name == "mean") key.op = OpId::Mean;
    else if (op_name == "sum") key.op = OpId::Sum;
    else if (op_name == "max") key.op = OpId::Max;
    else if (op_name == "cross_entropy_loss" || op_name == "cross_entropy") key.op = OpId::CrossEntropyLoss;
    else if (op_name == "mse_loss") key.op = OpId::MSELoss;
    else if (op_name == "l1_loss") key.op = OpId::L1Loss;
    else if (op_name == "matmul_q4_0") key.op = OpId::MatMulQ4_0;
    else if (op_name == "matmul_q4_K") key.op = OpId::MatMulQ4_K;
    else if (op_name == "matmul_q5_K") key.op = OpId::MatMulQ5_K;
    else if (op_name == "matmul_blas") {
        key.op = OpId::MatMulBLAS;
        key.provider = Provider::BLAS;
    } else {
        uint16_t hashed = static_cast<uint16_t>(10000 + (std::hash<std::string>{}(op_name) % 50000));
        key.op = static_cast<OpId>(hashed);
    }

    return key;
}

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry reg;
    return reg;
}

void KernelRegistry::register_kernel(KernelKey key, KernelFn fn) {
    registry_[key] = fn;
}

KernelFn KernelRegistry::lookup(KernelKey key) const {
    auto it = registry_.find(key);
    if (it != registry_.end()) {
        return it->second;
    }
    return nullptr;
}

KernelFn KernelRegistry::dispatch(KernelKey key) const {
    register_cpu_kernels();
    return lookup(key);
}

void KernelRegistry::register_kernel(const std::string& op_name, ISA isa, KernelFn fn) {
    if (isa == ISA::AVX2 && !has_avx2()) {
        return;
    }
    register_kernel(string_to_kernel_key(op_name), fn);
}

KernelFn KernelRegistry::lookup(const std::string& op_name, ISA isa) const {
    register_cpu_kernels();
    return lookup(string_to_kernel_key(op_name));
}

KernelFn KernelRegistry::dispatch(const std::string& op_name) const {
    register_cpu_kernels();
    return lookup(string_to_kernel_key(op_name));
}

void register_scalar_elementwise_kernels();
void register_avx2_elementwise_kernels();
void register_scalar_gemm_kernels();
void register_avx2_gemm_kernels();
void register_avx2_quants_kernels();
void register_blas_kernels();

static bool g_kernels_registered = false;

void register_cpu_kernels() {
    if (!g_kernels_registered) {
        g_kernels_registered = true;
        register_scalar_elementwise_kernels();
        if (has_avx2()) {
            register_avx2_elementwise_kernels();
            register_avx2_gemm_kernels();
            register_avx2_quants_kernels();
        } else {
            register_scalar_gemm_kernels();
        }
        register_blas_kernels();
    }
}

void KernelRegistry::clear() {
    registry_.clear();
    g_kernels_registered = false;
}

} // namespace axon::cpu
