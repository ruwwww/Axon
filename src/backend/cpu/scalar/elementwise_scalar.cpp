#include "axon/backend/registry.h"
#include "axon/backend/simd/vec.h"
#include "axon/backend/simd/scalar.h"
#include "axon/tensor/tensor_iterator.h"
#include <cmath>

namespace axon::cpu {

template<typename VecT, typename Op>
static Expected<void> elementwise_binary_kernel(KernelContext& ctx, Op op) {
    Tensor& out = ctx.outputs[0];
    const Tensor& a = ctx.inputs[0];
    const Tensor& b = ctx.inputs[1];

    TensorIterator<float> out_it(out);
    TensorIterator<const float> a_it(a);
    TensorIterator<const float> b_it(b);
    int64_t n = out.type().numel();

    for (int64_t i = 0; i < n; ++i) {
        out_it[i] = op(a_it[i], b_it[i]);
    }
    return {};
}

static Expected<void> add_scalar(KernelContext& ctx) {
    return elementwise_binary_kernel<simd::Vec<float, ISA::Scalar>>(ctx, [](float x, float y) { return x + y; });
}

static Expected<void> mul_scalar(KernelContext& ctx) {
    return elementwise_binary_kernel<simd::Vec<float, ISA::Scalar>>(ctx, [](float x, float y) { return x * y; });
}

static Expected<void> relu_scalar(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& x = ctx.inputs[0];
    TensorIterator<float> out_it(out);
    TensorIterator<const float> x_it(x);
    int64_t n = out.type().numel();
    for (int64_t i = 0; i < n; ++i) {
        out_it[i] = x_it[i] > 0.0f ? x_it[i] : 0.0f;
    }
    return {};
}

static Expected<void> gelu_scalar(KernelContext& ctx) {
    Tensor& out = ctx.outputs[0];
    const Tensor& x = ctx.inputs[0];
    TensorIterator<float> out_it(out);
    TensorIterator<const float> x_it(x);
    int64_t n = out.type().numel();
    constexpr float alpha = 0.79788456f;
    constexpr float beta = 0.044715f;
    for (int64_t i = 0; i < n; ++i) {
        float xi = x_it[i];
        float x3 = xi * xi * xi;
        out_it[i] = 0.5f * xi * (1.0f + std::tanh(alpha * (xi + beta * x3)));
    }
    return {};
}

void register_scalar_elementwise_kernels() {
    auto& reg = KernelRegistry::instance();
    reg.register_kernel({OpId::Add, Device::CPU, DType::Float32, Provider::AxonNative}, add_scalar);
    reg.register_kernel({OpId::Mul, Device::CPU, DType::Float32, Provider::AxonNative}, mul_scalar);
    reg.register_kernel({OpId::ReLU, Device::CPU, DType::Float32, Provider::AxonNative}, relu_scalar);
    reg.register_kernel({OpId::GELU, Device::CPU, DType::Float32, Provider::AxonNative}, gelu_scalar);
}

} // namespace axon::cpu
