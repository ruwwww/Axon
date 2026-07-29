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

// ── Conv2d ─────────────────────────────────────────────────────────────

Expected<void> conv2d(Tensor& out, const Tensor& input, const Tensor& weight,
                      int64_t stride, int64_t padding) {
    const auto& in_shape = input.type().shape();
    const auto& w_shape = weight.type().shape();
    const auto& out_shape = out.type().shape();

    if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
        return Error{"cpu::conv2d: all inputs must be 4D (N, C, H, W)"};
    }

    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OC = w_shape[0], IC = w_shape[1], KH = w_shape[2], KW = w_shape[3];
    auto OH = out_shape[2], OW = out_shape[3];

    if (C != IC) return Error{"cpu::conv2d: input channels != weight in_channels"};
    if (out_shape[0] != N || out_shape[1] != OC) return Error{"cpu::conv2d: output shape mismatch"};

    auto* inp = input.data<const float>();
    auto* w = weight.data<const float>();
    auto* o = out.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int64_t ic = 0; ic < IC; ++ic) {
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                int64_t ih = oh * stride + kh - padding;
                                int64_t iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += inp[n * C * H * W + ic * H * W + ih * W + iw]
                                         * w[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw];
                                }
                            }
                        }
                    }
                    o[n * OC * OH * OW + oc * OH * OW + oh * OW + ow] = sum;
                }
            }
        }
    }
    return {};
}

// ── MaxPool2d ──────────────────────────────────────────────────────────

Expected<void> maxpool2d(Tensor& out, const Tensor& input,
                         int64_t kernel, int64_t stride) {
    const auto& in_shape = input.type().shape();
    const auto& out_shape = out.type().shape();

    if (in_shape.size() != 4 || out_shape.size() != 4) {
        return Error{"cpu::maxpool2d: all inputs must be 4D (N, C, H, W)"};
    }

    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OH = out_shape[2], OW = out_shape[3];

    auto* inp = input.data<const float>();
    auto* o = out.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t kh = 0; kh < kernel; ++kh) {
                        for (int64_t kw = 0; kw < kernel; ++kw) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            if (ih < H && iw < W) {
                                float val = inp[n * C * H * W + c * H * W + ih * W + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    o[n * C * OH * OW + c * OH * OW + oh * OW + ow] = max_val;
                }
            }
        }
    }
    return {};
}

Expected<void> avgpool2d(Tensor& out, const Tensor& input,
                         int64_t kernel, int64_t stride) {
    const auto& in_shape = input.type().shape();
    const auto& out_shape = out.type().shape();

    if (in_shape.size() != 4 || out_shape.size() != 4) {
        return Error{"cpu::avgpool2d: all inputs must be 4D (N, C, H, W)"};
    }

    auto N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    auto OH = out_shape[2], OW = out_shape[3];

    auto* inp = input.data<const float>();
    auto* o = out.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int64_t kh = 0; kh < kernel; ++kh) {
                        for (int64_t kw = 0; kw < kernel; ++kw) {
                            int64_t ih = oh * stride + kh;
                            int64_t iw = ow * stride + kw;
                            if (ih < H && iw < W) {
                                sum += inp[n * C * H * W + c * H * W + ih * W + iw];
                            }
                        }
                    }
                    o[n * C * OH * OW + c * OH * OW + oh * OW + ow] = sum / static_cast<float>(kernel * kernel);
                }
            }
        }
    }
    return {};
}

// ── BatchNorm ──────────────────────────────────────────────────────────

Expected<void> batchnorm(Tensor& out, const Tensor& input,
                         const Tensor& gamma, const Tensor& beta,
                         const Tensor& running_mean, const Tensor& running_var,
                         float momentum, float epsilon, bool training) {
    const auto& in_shape = input.type().shape();
    int64_t N = in_shape[0], C = in_shape[1];
    int64_t spatial = 1;
    for (size_t i = 2; i < in_shape.size(); ++i) spatial *= in_shape[i];
    int64_t num_elements = N * spatial;

    auto* inp = input.data<const float>();
    auto* g = gamma.data<const float>();
    auto* b = beta.data<const float>();
    auto* rm = running_mean.data<const float>();
    auto* rv = running_var.data<const float>();
    auto* o = out.data<float>();

    if (training) {
        // Compute mean per channel
        std::vector<float> mean(C, 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int64_t s = 0; s < spatial; ++s) {
                    sum += inp[n * C * spatial + c * spatial + s];
                }
                mean[c] += sum;
            }
        }
        for (int64_t c = 0; c < C; ++c) mean[c] /= static_cast<float>(num_elements);

        // Compute variance per channel
        std::vector<float> var(C, 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int64_t s = 0; s < spatial; ++s) {
                    float diff = inp[n * C * spatial + c * spatial + s] - mean[c];
                    sum += diff * diff;
                }
                var[c] += sum;
            }
        }
        for (int64_t c = 0; c < C; ++c) var[c] /= static_cast<float>(num_elements);

        // Normalize
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float inv_std = 1.0f / std::sqrt(var[c] + epsilon);
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    o[idx] = (inp[idx] - mean[c]) * inv_std * g[c] + b[c];
                }
            }
        }

        // Update running stats (in-place, mutable through non-const pointers)
        // We need to cast away constness since Tensor::data returns const T*
        // but running_mean/var need updating
        auto* rm_mut = const_cast<float*>(rm);
        auto* rv_mut = const_cast<float*>(rv);
        for (int64_t c = 0; c < C; ++c) {
            rm_mut[c] = momentum * rm_mut[c] + (1.0f - momentum) * mean[c];
            rv_mut[c] = momentum * rv_mut[c] + (1.0f - momentum) * var[c];
        }
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float inv_std = 1.0f / std::sqrt(rv[c] + epsilon);
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    o[idx] = (inp[idx] - rm[c]) * inv_std * g[c] + b[c];
                }
            }
        }
    }
    return {};
}

// ── LayerNorm ──────────────────────────────────────────────────────────

Expected<void> layernorm(Tensor& out, const Tensor& input,
                         const Tensor& gamma, const Tensor& beta,
                         float epsilon) {
    const auto& in_shape = input.type().shape();
    if (in_shape.empty()) return Error{"cpu::layernorm: input must have at least 1 dimension"};

    auto N = in_shape[0];
    int64_t D = 1;
    for (size_t i = 1; i < in_shape.size(); ++i) D *= in_shape[i];

    auto* inp = input.data<const float>();
    auto* g = gamma.data<const float>();
    auto* b = beta.data<const float>();
    auto* o = out.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d) mean += inp[n * D + d];
        mean /= static_cast<float>(D);

        float variance = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float diff = inp[n * D + d] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(D);

        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int64_t d = 0; d < D; ++d) {
            o[n * D + d] = (inp[n * D + d] - mean) * inv_std * g[d] + b[d];
        }
    }
    return {};
}

} // namespace axon::cpu
