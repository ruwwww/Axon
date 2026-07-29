#include "axon/backend/cpu_backend.h"
#include <cmath>
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

Expected<void> matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.type().shape();
    const auto& b_shape = b.type().shape();
    const auto& out_shape = out.type().shape();

    if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2) {
        return Error{"cpu::matmul: all inputs must be 2D"};
    }

    auto M = a_shape[0];
    auto K = a_shape[1];
    auto N = b_shape[1];

    if (a_shape[1] != b_shape[0]) {
        return Error{"cpu::matmul: inner dimensions must match"};
    }
    if (out_shape[0] != M || out_shape[1] != N) {
        return Error{"cpu::matmul: output shape mismatch"};
    }
    if (a.type().dtype() != DType::Float32 || b.type().dtype() != DType::Float32 || out.type().dtype() != DType::Float32) {
        return Error{"cpu::matmul: only Float32 supported"};
    }

    auto* a_ptr = a.data<const float>();
    auto* b_ptr = b.data<const float>();
    auto* out_ptr = out.data<float>();

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_ptr[i * K + k] * b_ptr[k * N + j];
            }
            out_ptr[i * N + j] = sum;
        }
    }

    return {};
}

Expected<void> relu(Tensor& out, const Tensor& x) {
    if (x.type().shape() != out.type().shape()) {
        return Error{"cpu::relu: shape mismatch"};
    }
    if (x.type().dtype() != DType::Float32 || out.type().dtype() != DType::Float32) {
        return Error{"cpu::relu: only Float32 supported"};
    }

    auto* x_ptr = x.data<const float>();
    auto* out_ptr = out.data<float>();
    auto n = x.type().numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = x_ptr[i] > 0.0f ? x_ptr[i] : 0.0f;
    }

    return {};
}

Expected<void> log_softmax(Tensor& out, const Tensor& x) {
    if (x.type().shape() != out.type().shape()) {
        return Error{"cpu::log_softmax: shape mismatch"};
    }
    if (x.type().dtype() != DType::Float32 || out.type().dtype() != DType::Float32) {
        return Error{"cpu::log_softmax: only Float32 supported"};
    }

    const auto& shape = x.type().shape();
    int64_t n_rows = shape.size() == 2 ? shape[0] : 1;
    int64_t n_cols = shape.size() == 2 ? shape[1] : shape[0];

    auto* x_ptr = x.data<const float>();
    auto* out_ptr = out.data<float>();

    for (int64_t i = 0; i < n_rows; ++i) {
        float max_val = x_ptr[i * n_cols];
        for (int64_t j = 1; j < n_cols; ++j) {
            if (x_ptr[i * n_cols + j] > max_val) max_val = x_ptr[i * n_cols + j];
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < n_cols; ++j) {
            sum += std::exp(x_ptr[i * n_cols + j] - max_val);
        }
        float log_sum = std::log(sum);

        for (int64_t j = 0; j < n_cols; ++j) {
            out_ptr[i * n_cols + j] = x_ptr[i * n_cols + j] - max_val - log_sum;
        }
    }

    return {};
}

Expected<void> softmax(Tensor& out, const Tensor& x) {
    if (x.type().shape() != out.type().shape()) {
        return Error{"cpu::softmax: shape mismatch"};
    }
    if (x.type().dtype() != DType::Float32 || out.type().dtype() != DType::Float32) {
        return Error{"cpu::softmax: only Float32 supported"};
    }

    const auto& shape = x.type().shape();
    int64_t n_rows = shape.size() == 2 ? shape[0] : 1;
    int64_t n_cols = shape.size() == 2 ? shape[1] : shape[0];

    auto* x_ptr = x.data<const float>();
    auto* out_ptr = out.data<float>();

    for (int64_t i = 0; i < n_rows; ++i) {
        float max_val = x_ptr[i * n_cols];
        for (int64_t j = 1; j < n_cols; ++j) {
            if (x_ptr[i * n_cols + j] > max_val) max_val = x_ptr[i * n_cols + j];
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < n_cols; ++j) {
            out_ptr[i * n_cols + j] = std::exp(x_ptr[i * n_cols + j] - max_val);
            sum += out_ptr[i * n_cols + j];
        }

        for (int64_t j = 0; j < n_cols; ++j) {
            out_ptr[i * n_cols + j] /= sum;
        }
    }

    return {};
}

} // namespace axon::cpu
