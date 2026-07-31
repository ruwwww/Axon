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

static Expected<void> matmul_q4_K_avx2(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];

    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const axon::block_q4_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d = axon::half_to_float(block.d);
                float min = axon::half_to_float(block.dmin);

                const uint8_t* q = block.qs;
                int is = 0;
                int k_base = static_cast<int>(bk * 256);

                for (int b_sub = 0; b_sub < 256; b_sub += 64) {
                    uint8_t sc, m;
                    axon::get_scale_min_k4(is, block.scales, &sc, &m);
                    float d1 = d * sc;
                    float m1 = min * m;
                    axon::get_scale_min_k4(is + 1, block.scales, &sc, &m);
                    float d2 = d * sc;
                    float m2 = min * m;

                    for (int l = 0; l < 32; ++l) {
                        int k_idx = k_base + b_sub + l;
                        if (k_idx < K) {
                            float val = d1 * static_cast<float>(q[l] & 0xF) - m1;
                            sum += val * b_ptr[k_idx * N + j];
                        }
                    }
                    for (int l = 0; l < 32; ++l) {
                        int k_idx = k_base + b_sub + 32 + l;
                        if (k_idx < K) {
                            float val = d2 * static_cast<float>(q[l] >> 4) - m2;
                            sum += val * b_ptr[k_idx * N + j];
                        }
                    }
                    q += 32;
                    is += 2;
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }
    return {};
}

static Expected<void> matmul_q5_K_avx2(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];

    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const axon::block_q5_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d = axon::half_to_float(block.d);
                float min = axon::half_to_float(block.dmin);

                const uint8_t* ql = block.qs;
                const uint8_t* qh = block.qh;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                int k_base = static_cast<int>(bk * 256);

                for (int b_sub = 0; b_sub < 256; b_sub += 64) {
                    uint8_t sc, m;
                    axon::get_scale_min_k4(is, block.scales, &sc, &m);
                    float d1 = d * sc;
                    float m1 = min * m;
                    axon::get_scale_min_k4(is + 1, block.scales, &sc, &m);
                    float d2 = d * sc;
                    float m2 = min * m;

                    for (int l = 0; l < 32; ++l) {
                        int k_idx = k_base + b_sub + l;
                        if (k_idx < K) {
                            uint8_t q_val = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                            float val = d1 * static_cast<float>(q_val) - m1;
                            sum += val * b_ptr[k_idx * N + j];
                        }
                    }
                    for (int l = 0; l < 32; ++l) {
                        int k_idx = k_base + b_sub + 32 + l;
                        if (k_idx < K) {
                            uint8_t q_val = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                            float val = d2 * static_cast<float>(q_val) - m2;
                            sum += val * b_ptr[k_idx * N + j];
                        }
                    }
                    ql += 32;
                    u1 <<= 2;
                    u2 <<= 2;
                    is += 2;
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
    reg.register_kernel("matmul_q4_K", ISA::AVX2, matmul_q4_K_avx2);
    reg.register_kernel("matmul_q5_K", ISA::AVX2, matmul_q5_K_avx2);
}

#endif

} // namespace axon::cpu
