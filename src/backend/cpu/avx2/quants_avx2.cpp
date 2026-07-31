#include "axon/backend/registry.h"
#include "axon/backend/simd/vec.h"
#include "axon/backend/simd/avx2.h"
#include "axon/storage/quantization.h"
#include "axon/tensor/tensor.h"

namespace axon::cpu {

#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86)))

using Vec8f = simd::Vec<float, ISA::AVX2>;

static Expected<void> matmul_q4_0_avx2(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];

    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const axon::block_q4_0*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d = axon::half_to_float(block.d);

                int k_base = static_cast<int>(bk * 32);

                for (int l = 0; l < 16; ++l) {
                    int k_lo = k_base + l * 2;
                    int k_hi = k_base + l * 2 + 1;

                    if (k_lo < K) {
                        int8_t q_lo = static_cast<int8_t>(block.qs[l] & 0x0F) - 8;
                        sum += d * static_cast<float>(q_lo) * b_ptr[k_lo * N + j];
                    }
                    if (k_hi < K) {
                        int8_t q_hi = static_cast<int8_t>(block.qs[l] >> 4) - 8;
                        sum += d * static_cast<float>(q_hi) * b_ptr[k_hi * N + j];
                    }
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }
    return {};
}

void register_avx2_quants_kernels() {
    auto& reg = KernelRegistry::instance();
    reg.register_kernel("matmul_q4_0", ISA::AVX2, matmul_q4_0_avx2);
}

#endif

} // namespace axon::cpu
