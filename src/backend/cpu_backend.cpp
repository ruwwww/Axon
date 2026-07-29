#include "axon/backend/cpu_backend.h"
#include <string>

namespace axon::cpu {

static Expected<void> validate_same_shape(const Tensor& a, const Tensor& b, const Tensor& out) {
    if (a.type().shape() != b.type().shape() || a.type().shape() != out.type().shape()) {
        return Error{"cpu: shape mismatch in element-wise operation"};
    }
    return {};
}

template <typename Op>
static Expected<void> elementwise_op(Tensor& out, const Tensor& a, const Tensor& b, Op op) {
    auto check = validate_same_shape(a, b, out);
    if (!check) return check.error();

    if (a.type().dtype() == DType::Float32 && b.type().dtype() == DType::Float32 && out.type().dtype() == DType::Float32) {
        auto* a_ptr = a.data<const float>();
        auto* b_ptr = b.data<const float>();
        auto* out_ptr = out.data<float>();
        auto n = a.type().numel();
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = op(a_ptr[i], b_ptr[i]);
        }
        return {};
    }

    return Error{"cpu: unsupported dtype for element-wise operation"};
}

Expected<void> add(Tensor& out, const Tensor& a, const Tensor& b) {
    return elementwise_op(out, a, b, [](float x, float y) { return x + y; });
}

Expected<void> sub(Tensor& out, const Tensor& a, const Tensor& b) {
    return elementwise_op(out, a, b, [](float x, float y) { return x - y; });
}

Expected<void> mul(Tensor& out, const Tensor& a, const Tensor& b) {
    return elementwise_op(out, a, b, [](float x, float y) { return x * y; });
}

Expected<void> div(Tensor& out, const Tensor& a, const Tensor& b) {
    return elementwise_op(out, a, b, [](float x, float y) { return x / y; });
}

} // namespace axon::cpu
