#include "axon/backend/registry.h"
#include "axon/tensor/tensor_iterator.h"

namespace axon::cpu {

static Expected<void> matmul_scalar(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    const auto& a_shape = a.type().shape();
    const auto& b_shape = b.type().shape();

    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t N = b_shape[1];

    if (out.type().is_contiguous() && a.type().is_contiguous() && b.type().is_contiguous()) {
        float* out_ptr = out.data<float>();
        const float* a_ptr = a.data<const float>();
        const float* b_ptr = b.data<const float>();

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                }
                out_ptr[i * N + j] = sum;
            }
        }
    } else {
        TensorIterator<const float> a_it(a);
        TensorIterator<const float> b_it(b);
        TensorIterator<float> out_it(out);

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += a_it[i * K + k] * b_it[k * N + j];
                }
                out_it[i * N + j] = sum;
            }
        }
    }

    return {};
}

void register_scalar_gemm_kernels() {
    auto& reg = KernelRegistry::instance();
    reg.register_kernel("matmul", ISA::Scalar, matmul_scalar);
}

} // namespace axon::cpu
