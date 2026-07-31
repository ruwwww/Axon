#include "axon/backend/registry.h"
#include "axon/backend/simd/vec.h"
#include "axon/backend/simd/avx2.h"
#include "axon/tensor/tensor_iterator.h"
#include <cmath>

namespace axon::cpu {

#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86)))

using Vec8f = simd::Vec<float, ISA::AVX2>;

static Expected<void> add_avx2(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    auto n = out.type().numel();

    if (out.type().is_contiguous() && a.type().is_contiguous() && b.type().is_contiguous()) {
        float* out_ptr = out.data<float>();
        const float* a_ptr = a.data<const float>();
        const float* b_ptr = b.data<const float>();

        int64_t i = 0;
        for (; i <= n - Vec8f::size; i += Vec8f::size) {
            auto va = Vec8f::load(a_ptr + i);
            auto vb = Vec8f::load(b_ptr + i);
            (va + vb).store(out_ptr + i);
        }
        for (; i < n; ++i) {
            out_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    } else {
        TensorIterator<float> out_it(out);
        TensorIterator<const float> a_it(a);
        TensorIterator<const float> b_it(b);
        for (int64_t i = 0; i < n; ++i) {
            out_it[i] = a_it[i] + b_it[i];
        }
    }

    return {};
}

static Expected<void> mul_avx2(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    auto n = out.type().numel();

    if (out.type().is_contiguous() && a.type().is_contiguous() && b.type().is_contiguous()) {
        float* out_ptr = out.data<float>();
        const float* a_ptr = a.data<const float>();
        const float* b_ptr = b.data<const float>();

        int64_t i = 0;
        for (; i <= n - Vec8f::size; i += Vec8f::size) {
            auto va = Vec8f::load(a_ptr + i);
            auto vb = Vec8f::load(b_ptr + i);
            (va * vb).store(out_ptr + i);
        }
        for (; i < n; ++i) {
            out_ptr[i] = a_ptr[i] * b_ptr[i];
        }
    } else {
        TensorIterator<float> out_it(out);
        TensorIterator<const float> a_it(a);
        TensorIterator<const float> b_it(b);
        for (int64_t i = 0; i < n; ++i) {
            out_it[i] = a_it[i] * b_it[i];
        }
    }

    return {};
}

static Expected<void> relu_avx2(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& x = ctx.inputs[0];

    auto n = out.type().numel();
    auto vzero = Vec8f::set1(0.0f);

    if (out.type().is_contiguous() && x.type().is_contiguous()) {
        float* out_ptr = out.data<float>();
        const float* x_ptr = x.data<const float>();

        int64_t i = 0;
        for (; i <= n - Vec8f::size; i += Vec8f::size) {
            auto vx = Vec8f::load(x_ptr + i);
            vx.max(vzero).store(out_ptr + i);
        }
        for (; i < n; ++i) {
            out_ptr[i] = x_ptr[i] > 0.0f ? x_ptr[i] : 0.0f;
        }
    } else {
        TensorIterator<float> out_it(out);
        TensorIterator<const float> x_it(x);
        for (int64_t i = 0; i < n; ++i) {
            out_it[i] = x_it[i] > 0.0f ? x_it[i] : 0.0f;
        }
    }

    return {};
}

void register_avx2_elementwise_kernels() {
    auto& reg = KernelRegistry::instance();
    reg.register_kernel("add", ISA::AVX2, add_avx2);
    reg.register_kernel("mul", ISA::AVX2, mul_avx2);
    reg.register_kernel("relu", ISA::AVX2, relu_avx2);
}

#endif

} // namespace axon::cpu
