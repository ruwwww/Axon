#include "axon/backend/registry.h"
#include "axon/tensor/tensor.h"

#if defined(AXON_HAS_BLAS)
#include <cblas.h>
#endif

namespace axon::cpu {

#if defined(AXON_HAS_BLAS)

static Expected<void> matmul_blas(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    auto M = static_cast<int>(a.type().shape()[0]);
    auto K = static_cast<int>(a.type().shape()[1]);
    auto N = static_cast<int>(b.type().shape()[1]);

    const float* a_ptr = a.data<const float>();
    const float* b_ptr = b.data<const float>();
    float* o_ptr = out.data<float>();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, a_ptr, K, b_ptr, N, 0.0f, o_ptr, N);
    return {};
}

void register_blas_kernels() {
    auto& reg = KernelRegistry::instance();
    reg.register_kernel({OpId::MatMulBLAS, Device::CPU, DType::Float32, Provider::BLAS}, matmul_blas);
}

#else

void register_blas_kernels() {}

#endif

} // namespace axon::cpu
