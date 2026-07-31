#include "axon/backend/cpu_backend.h"
#include "axon/backend/registry.h"
#include "axon/tensor/tensor_iterator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
        TensorIterator<const float> a_it(a);
        TensorIterator<const float> b_it(b);
        TensorIterator<float> out_it(out);
        auto n = a.type().numel();
        for (int64_t i = 0; i < n; ++i) {
            out_it[i] = op(a_it[i], b_it[i]);
        }
        return {};
    }

    return Error{"cpu: unsupported dtype for element-wise operation"};
}

Expected<void> add(Tensor& out, const Tensor& a, const Tensor& b) {
    auto check = validate_same_shape(a, b, out);
    if (!check) return check.error();

    auto fn = KernelRegistry::instance().dispatch("add");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {a, b};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }
    return elementwise_op(out, a, b, [](float x, float y) { return x + y; });
}

Expected<void> sub(Tensor& out, const Tensor& a, const Tensor& b) {
    return elementwise_op(out, a, b, [](float x, float y) { return x - y; });
}

Expected<void> mul(Tensor& out, const Tensor& a, const Tensor& b) {
    auto check = validate_same_shape(a, b, out);
    if (!check) return check.error();

    auto fn = KernelRegistry::instance().dispatch("mul");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {a, b};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }
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

    auto fn = KernelRegistry::instance().dispatch("matmul");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {a, b};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }

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

    return {};
}

Expected<void> relu(Tensor& out, const Tensor& x) {
    if (x.type().shape() != out.type().shape()) {
        return Error{"cpu::relu: shape mismatch"};
    }
    if (x.type().dtype() != DType::Float32 || out.type().dtype() != DType::Float32) {
        return Error{"cpu::relu: only Float32 supported"};
    }

    auto fn = KernelRegistry::instance().dispatch("relu");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {x};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }

    TensorIterator<const float> x_it(x);
    TensorIterator<float> out_it(out);
    auto n = x.type().numel();

    for (int64_t i = 0; i < n; ++i) {
        out_it[i] = x_it[i] > 0.0f ? x_it[i] : 0.0f;
    }

    return {};
}

Expected<void> gelu(Tensor& out, const Tensor& x) {
    if (x.type().shape() != out.type().shape()) {
        return Error{"cpu::gelu: shape mismatch"};
    }
    if (x.type().dtype() != DType::Float32 || out.type().dtype() != DType::Float32) {
        return Error{"cpu::gelu: only Float32 supported"};
    }

    TensorIterator<const float> x_it(x);
    TensorIterator<float> out_it(out);
    auto n = x.type().numel();
    constexpr float alpha = 0.79788456f;  // sqrt(2/pi)
    constexpr float beta = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        float xi = x_it[i];
        float x3 = xi * xi * xi;
        float inner = alpha * (xi + beta * x3);
        out_it[i] = 0.5f * xi * (1.0f + std::tanh(inner));
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

    TensorIterator<const float> x_it(x);
    TensorIterator<float> out_it(out);

    for (int64_t i = 0; i < n_rows; ++i) {
        float max_val = x_it[i * n_cols];
        for (int64_t j = 1; j < n_cols; ++j) {
            float val = x_it[i * n_cols + j];
            if (val > max_val) max_val = val;
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < n_cols; ++j) {
            sum += std::exp(x_it[i * n_cols + j] - max_val);
        }
        float log_sum = std::log(sum);

        for (int64_t j = 0; j < n_cols; ++j) {
            out_it[i * n_cols + j] = x_it[i * n_cols + j] - max_val - log_sum;
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

    TensorIterator<const float> x_it(x);
    TensorIterator<float> out_it(out);

    for (int64_t i = 0; i < n_rows; ++i) {
        float max_val = x_it[i * n_cols];
        for (int64_t j = 1; j < n_cols; ++j) {
            float val = x_it[i * n_cols + j];
            if (val > max_val) max_val = val;
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < n_cols; ++j) {
            out_it[i * n_cols + j] = std::exp(x_it[i * n_cols + j] - max_val);
            sum += out_it[i * n_cols + j];
        }

        for (int64_t j = 0; j < n_cols; ++j) {
            out_it[i * n_cols + j] /= sum;
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

    TensorIterator<const float> inp_it(input);
    TensorIterator<const float> w_it(weight);
    TensorIterator<float> o_it(out);

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
                                    sum += inp_it[n * C * H * W + ic * H * W + ih * W + iw]
                                         * w_it[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw];
                                }
                            }
                        }
                    }
                    o_it[n * OC * OH * OW + oc * OH * OW + oh * OW + ow] = sum;
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

    TensorIterator<const float> inp_it(input);
    TensorIterator<const float> g_it(gamma);
    TensorIterator<const float> b_it(beta);
    TensorIterator<float> rm_it(running_mean);
    TensorIterator<float> rv_it(running_var);
    TensorIterator<float> o_it(out);

    if (training) {
        // Compute mean per channel
        std::vector<float> mean(C, 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int64_t s = 0; s < spatial; ++s) {
                    sum += inp_it[n * C * spatial + c * spatial + s];
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
                    float diff = inp_it[n * C * spatial + c * spatial + s] - mean[c];
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
                    o_it[idx] = (inp_it[idx] - mean[c]) * inv_std * g_it[c] + b_it[c];
                }
            }
        }

        // Update running stats
        for (int64_t c = 0; c < C; ++c) {
            rm_it[c] = momentum * rm_it[c] + (1.0f - momentum) * mean[c];
            rv_it[c] = momentum * rv_it[c] + (1.0f - momentum) * var[c];
        }
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                float inv_std = 1.0f / std::sqrt(rv_it[c] + epsilon);
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    o_it[idx] = (inp_it[idx] - rm_it[c]) * inv_std * g_it[c] + b_it[c];
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

    TensorIterator<const float> inp_it(input);
    TensorIterator<const float> g_it(gamma);
    TensorIterator<const float> b_it(beta);
    TensorIterator<float> o_it(out);

    for (int64_t n = 0; n < N; ++n) {
        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d) mean += inp_it[n * D + d];
        mean /= static_cast<float>(D);

        float variance = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float diff = inp_it[n * D + d] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(D);

        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int64_t d = 0; d < D; ++d) {
            o_it[n * D + d] = (inp_it[n * D + d] - mean) * inv_std * g_it[d] + b_it[d];
        }
    }
    return {};
}

// ── reduce_mean ─────────────────────────────────────────────────────────

Expected<void> reduce_mean(Tensor& out, const Tensor& input, const std::vector<int64_t>& dims) {
    const auto& shape = input.type().shape();
    auto ndim = shape.size();

    std::vector<bool> is_reduced(ndim, false);
    int64_t reduction_size = 1;
    for (auto d : dims) {
        if (d < 0 || static_cast<size_t>(d) >= ndim)
            return Error{"cpu::reduce_mean: dim out of range"};
        is_reduced[d] = true;
        reduction_size *= shape[d];
    }

    TensorIterator<const float> inp_it(input);
    TensorIterator<float> o_it(out);
    auto numel = input.type().numel();
    auto out_numel = out.type().numel();

    std::fill(out.data<float>(), out.data<float>() + out_numel, 0.0f);

    std::vector<int64_t> idx(ndim, 0);
    for (int64_t flat = 0; flat < numel; ++flat) {
        int64_t tmp = flat;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            idx[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        int64_t out_flat = 0;
        for (size_t d = 0; d < ndim; ++d) {
            if (!is_reduced[d]) {
                out_flat = out_flat * shape[d] + idx[d];
            }
        }
        o_it[out_flat] += inp_it[flat];
    }

    float inv = 1.0f / static_cast<float>(reduction_size);
    for (int64_t i = 0; i < out_numel; ++i) o_it[i] *= inv;
    return {};
}

// ── Half-precision helpers ──────────────────────────────────────────────

static inline uint16_t float_to_half(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xFF) - 127;
    uint32_t mantissa = u & 0x007FFFFF;

    if (exp > 15) return static_cast<uint16_t>(sign | 0x7C00);
    if (exp < -14) return static_cast<uint16_t>(sign);
    uint16_t h_exp = static_cast<uint16_t>((exp + 15) & 0x1F);
    uint16_t h_mant = static_cast<uint16_t>(mantissa >> 13);
    return static_cast<uint16_t>(sign | (h_exp << 10) | h_mant);
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>((h >> 15) & 0x1);
    int32_t exp = static_cast<int32_t>((h >> 10) & 0x1F);
    uint32_t mantissa = static_cast<uint32_t>(h & 0x03FF);

    if (exp == 0) {
        uint32_t u = sign << 31;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
    if (exp == 31) {
        uint32_t u = (sign << 31) | 0x7F800000 | (mantissa << 13);
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }

    uint32_t f_exp = static_cast<uint32_t>((exp - 15 + 127) & 0xFF);
    uint32_t u = (sign << 31) | (f_exp << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ── Block structures ───────────────────────────────────────────────────

struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};

struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};

// K-quant block structures (256-element blocks, GGML-compatible layout)

#pragma pack(push, 1)
struct block_q2_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[16];
    uint8_t qs[64];
};

struct block_q3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// Shared helper: decode 6-bit scale/min from 12-byte K-quant scales array (GGML get_scale_min_k4)
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ── Quantized size ─────────────────────────────────────────────────────

size_t quantized_size(size_t num_elements, QuantFormat format) {
    size_t num_blocks_32 = (num_elements + 31) / 32;
    size_t num_blocks_256 = (num_elements + 255) / 256;
    switch (format) {
        case QuantFormat::Q8_0: return num_blocks_32 * sizeof(block_q8_0);
        case QuantFormat::Q4_0: return num_blocks_32 * sizeof(block_q4_0);
        case QuantFormat::Q2_K: return num_blocks_256 * sizeof(block_q2_K);
        case QuantFormat::Q3_K: return num_blocks_256 * sizeof(block_q3_K);
        case QuantFormat::Q4_K: return num_blocks_256 * sizeof(block_q4_K);
        case QuantFormat::Q5_K: return num_blocks_256 * sizeof(block_q5_K);
        case QuantFormat::Q6_K: return num_blocks_256 * sizeof(block_q6_K);
        default: return 0;
    }
}

size_t quantized_size_2d(int64_t M, int64_t K, QuantFormat format) {
    switch (format) {
        case QuantFormat::Q8_0:
        case QuantFormat::Q4_0: {
            size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;
            size_t total_blocks = static_cast<size_t>(M) * blocks_per_row;
            if (format == QuantFormat::Q8_0) return total_blocks * sizeof(block_q8_0);
            return total_blocks * sizeof(block_q4_0);
        }
        case QuantFormat::Q2_K:
        case QuantFormat::Q3_K:
        case QuantFormat::Q4_K:
        case QuantFormat::Q5_K:
        case QuantFormat::Q6_K: {
            size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;
            size_t total_blocks = static_cast<size_t>(M) * blocks_per_row;
            switch (format) {
                case QuantFormat::Q2_K: return total_blocks * sizeof(block_q2_K);
                case QuantFormat::Q3_K: return total_blocks * sizeof(block_q3_K);
                case QuantFormat::Q4_K: return total_blocks * sizeof(block_q4_K);
                case QuantFormat::Q5_K: return total_blocks * sizeof(block_q5_K);
                case QuantFormat::Q6_K: return total_blocks * sizeof(block_q6_K);
                default: return 0;
            }
        }
        default: return 0;
    }
}

// ── Quantize helpers ───────────────────────────────────────────────────

static void quantize_block_q8_0(block_q8_0& block, const float* src, int count) {
    float amax = 0.0f;
    for (int i = 0; i < count; ++i) amax = std::max(amax, std::abs(src[i]));
    float d = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
    block.d = float_to_half(d);
    for (int i = 0; i < 32; ++i) {
        if (i < count) {
            float q = src[i] / d;
            block.qs[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(q))));
        } else {
            block.qs[i] = 0;
        }
    }
}

static void quantize_block_q4_0(block_q4_0& block, const float* src, int count) {
    float amax = 0.0f;
    for (int i = 0; i < count; ++i) amax = std::max(amax, std::abs(src[i]));
    float d = (amax == 0.0f) ? 1.0f : (amax / 8.0f);
    block.d = float_to_half(d);
    for (int i = 0; i < 16; ++i) {
        int lo_val = 0, hi_val = 0;
        int idx_lo = i * 2;
        int idx_hi = i * 2 + 1;
        if (idx_lo < count) {
            float q = src[idx_lo] / d;
            lo_val = std::max(-8, std::min(7, static_cast<int>(std::round(q))));
        }
        if (idx_hi < count) {
            float q = src[idx_hi] / d;
            hi_val = std::max(-8, std::min(7, static_cast<int>(std::round(q))));
        }
        block.qs[i] = static_cast<uint8_t>((lo_val + 8) | ((hi_val + 8) << 4));
    }
}

static void quantize_block_q2_K(block_q2_K& block, const float* src, int count) {
    int n_sub = 16;
    int sub_size = 16;
    float max_scale = 0.0f, max_min = 0.0f;
    float sub_scales[16], sub_mins[16];

    for (int j = 0; j < n_sub; ++j) {
        if (j * sub_size >= count) {
            sub_scales[j] = 0.0f;
            sub_mins[j] = 0.0f;
            continue;
        }
        float sub_min = src[j * sub_size];
        float sub_max = src[j * sub_size];
        for (int i = 1; i < sub_size && j * sub_size + i < count; ++i) {
            float v = src[j * sub_size + i];
            if (v < sub_min) sub_min = v;
            if (v > sub_max) sub_max = v;
        }
        float range = sub_max - sub_min;
        if (range < 1e-10f) range = 1.0f;
        sub_scales[j] = range / 3.0f;
        float clamped_min = (sub_min < 0) ? sub_min : 0;
        sub_mins[j] = -clamped_min;
        if (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        if (sub_mins[j] > max_min) max_min = sub_mins[j];
    }

    // Quantize sub-block scales and mins to 4 bits
    float iscale_d = (max_scale > 0.0f) ? 15.0f / max_scale : 0.0f;
    float iscale_m = (max_min > 0.0f) ? 15.0f / max_min : 0.0f;
    block.d = float_to_half(max_scale / 15.0f);
    block.dmin = float_to_half(max_min / 15.0f);

    for (int j = 0; j < n_sub; ++j) {
        int qs = std::min(15, std::max(0, static_cast<int>(std::round(sub_scales[j] * iscale_d))));
        int qm = std::min(15, std::max(0, static_cast<int>(std::round(sub_mins[j] * iscale_m))));
        block.scales[j] = static_cast<uint8_t>(qs | (qm << 4));
    }

    // Re-quantize elements with actual scale/min
    uint8_t L[256];
    float d_vals[16], m_vals[16];
    float d_super = half_to_float(block.d);
    float m_super = half_to_float(block.dmin);
    for (int j = 0; j < n_sub; ++j) {
        d_vals[j] = d_super * (block.scales[j] & 0xF);
        m_vals[j] = m_super * (block.scales[j] >> 4);
    }
    for (int j = 0; j < n_sub; ++j) {
        for (int i = 0; i < sub_size && j * sub_size + i < count; ++i) {
            float val = src[j * sub_size + i];
            int l = static_cast<int>(std::round((val + m_vals[j]) / d_vals[j]));
            l = std::max(0, std::min(3, l));
            L[j * sub_size + i] = static_cast<uint8_t>(l);
        }
        for (int i = std::max(0, count - j * sub_size); i < sub_size; ++i) {
            L[j * sub_size + i] = 0;
        }
    }

    // Pack 4 × 2-bit values per byte, 128 elements at a time (GGML batch layout)
    memset(block.qs, 0, 64);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int v0 = (n + l < count) ? (L[n + l] & 3) : 0;
            int v1 = (n + l + 32 < count) ? ((L[n + l + 32] & 3) << 2) : 0;
            int v2 = (n + l + 64 < count) ? ((L[n + l + 64] & 3) << 4) : 0;
            int v3 = (n + l + 96 < count) ? ((L[n + l + 96] & 3) << 6) : 0;
            block.qs[n / 4 + l] = static_cast<uint8_t>(v0 | v1 | v2 | v3);
        }
    }
}

static void quantize_block_q3_K(block_q3_K& block, const float* src, int count) {
    int n_sub = 16;
    int sub_size = 16;
    float max_abs_scale = 0.0f;
    float sub_scales[16];

    // Per sub-block: compute max absolute value and scale
    for (int j = 0; j < n_sub; ++j) {
        float amax = 0.0f;
        for (int i = 0; i < sub_size && j * sub_size + i < count; ++i) {
            amax = std::max(amax, std::abs(src[j * sub_size + i]));
        }
        sub_scales[j] = (amax < 1e-10f) ? 0.0f : (amax / 4.0f);
        float abs_s = std::abs(sub_scales[j]);
        if (abs_s > max_abs_scale) max_abs_scale = abs_s;
    }

    // Quantize scales to 6-bit (0..63), centered at 32
    memset(block.scales, 0, 12);
    if (max_abs_scale > 0.0f) {
        float iscale = -32.0f / max_abs_scale;
        for (int j = 0; j < n_sub; ++j) {
            int l = static_cast<int>(std::round(iscale * sub_scales[j]));
            l = std::max(-32, std::min(31, l)) + 32; // 0..63
            // Pack into the 12-byte Q3_K scales format
            if (j < 8) {
                block.scales[j] = l & 0xF;
            } else {
                block.scales[j - 8] |= ((l & 0xF) << 4);
            }
            l >>= 4;
            block.scales[j % 4 + 8] |= (l << (2 * (j / 4)));
        }
        block.d = float_to_half(1.0f / iscale);
    } else {
        block.d = float_to_half(0.0f);
    }

    // Quantize elements to -4..3
    int8_t L[256];
    float d_all = half_to_float(block.d);
    for (int j = 0; j < n_sub; ++j) {
        int sc;
        if (j < 8) sc = block.scales[j] & 0xF;
        else sc = block.scales[j - 8] >> 4;
        sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
        float d_sub = d_all * static_cast<float>(sc);
        if (std::abs(d_sub) < 1e-10f) d_sub = 1.0f;
        for (int i = 0; i < sub_size && j * sub_size + i < count; ++i) {
            int l = static_cast<int>(std::round(src[j * sub_size + i] / d_sub));
            l = std::max(-4, std::min(3, l));
            L[j * sub_size + i] = static_cast<int8_t>(l);
        }
        for (int i = std::max(0, count - j * sub_size); i < sub_size; ++i) {
            L[j * sub_size + i] = 0;
        }
    }

    // Pack: low 2 bits in qs, high bit in hmask
    // hmask: 32 bytes, each byte holds 8 bits (one per 32-element block)
    memset(block.hmask, 0, 32);
    memset(block.qs, 0, 64);
    int m_idx = 0;
    uint8_t hm = 1;
    for (int j = 0; j < 256 && j < count; ++j) {
        int v = L[j] + 4; // shift to 0..7
        if (v > 3) {
            block.hmask[m_idx] |= hm;
            v -= 4;
        }
        if (++m_idx == 32) { m_idx = 0; hm <<= 1; }
    }
    // qs: 4 × 2-bit values per byte, GGML batch layout
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            block.qs[n / 4 + l] = static_cast<uint8_t>(
                (L[n + l] & 3) |
                ((L[n + l + 32] & 3) << 2) |
                ((L[n + l + 64] & 3) << 4) |
                ((L[n + l + 96] & 3) << 6)
            );
        }
    }
}

static void quantize_block_q4_K(block_q4_K& block, const float* src, int count) {
    int n_groups = 8;
    int group_size = 32;
    float max_scale = 0.0f, max_min = 0.0f;
    float scales[8], mins[8];

    for (int j = 0; j < n_groups; ++j) {
        if (j * group_size >= count) {
            scales[j] = 0.0f;
            mins[j] = 0.0f;
            continue;
        }
        float g_min = src[j * group_size], g_max = src[j * group_size];
        for (int i = 1; i < group_size && j * group_size + i < count; ++i) {
            float v = src[j * group_size + i];
            if (v < g_min) g_min = v;
            if (v > g_max) g_max = v;
        }
        float range = g_max - g_min;
        if (range < 1e-10f) range = 1.0f;
        scales[j] = range / 15.0f;
        float clamped_min = (g_min < 0) ? g_min : 0;
        mins[j] = -clamped_min;
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    // Quantize scales and mins to 6 bits
    float iscale = (max_scale > 0.0f) ? 63.0f / max_scale : 0.0f;
    float iscale_m = (max_min > 0.0f) ? 63.0f / max_min : 0.0f;
    block.d = float_to_half(max_scale / 63.0f);
    block.dmin = float_to_half(max_min / 63.0f);

    // Pack into 12-byte format (GGML get_scale_min_k4 compatible)
    memset(block.scales, 0, 12);
    for (int j = 0; j < n_groups; ++j) {
        uint8_t ls = std::min(63, std::max(0, static_cast<int>(std::round(scales[j] * iscale))));
        uint8_t lm = std::min(63, std::max(0, static_cast<int>(std::round(mins[j] * iscale_m))));
        if (j < 4) {
            block.scales[j] = ls;
            block.scales[j + 4] = lm;
        } else {
            block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            block.scales[j - 4] |= ((ls >> 4) << 6);
            block.scales[j - 0] |= ((lm >> 4) << 6);
        }
    }

    // Re-quantize elements
    uint8_t L[256];
    float d_super = half_to_float(block.d);
    float m_super = half_to_float(block.dmin);
    for (int j = 0; j < n_groups; ++j) {
        uint8_t sc, m;
        get_scale_min_k4(j, block.scales, &sc, &m);
        float d_eff = d_super * sc;
        float m_eff = m_super * m;
        if (d_eff < 1e-10f) d_eff = 1.0f;
        for (int i = 0; i < group_size && j * group_size + i < count; ++i) {
            int l = static_cast<int>(std::round((src[j * group_size + i] + m_eff) / d_eff));
            L[j * group_size + i] = static_cast<uint8_t>(std::max(0, std::min(15, l)));
        }
        for (int i = std::max(0, count - j * group_size); i < group_size; ++i) {
            L[j * group_size + i] = 0;
        }
    }

    // Pack 2 × 4-bit per byte, 64 elements at a time
    uint8_t* q = block.qs;
    for (int j = 0; j < 256; j += 64) {
        for (int l = 0; l < 32; ++l) {
            q[l] = L[j + l] | (L[j + l + 32] << 4);
        }
        q += 32;
    }
}

static void quantize_block_q5_K(block_q5_K& block, const float* src, int count) {
    int n_groups = 8;
    int group_size = 32;
    float max_scale = 0.0f, max_min = 0.0f;
    float scales[8], mins[8];

    for (int j = 0; j < n_groups; ++j) {
        if (j * group_size >= count) {
            scales[j] = 0.0f;
            mins[j] = 0.0f;
            continue;
        }
        float g_min = src[j * group_size], g_max = src[j * group_size];
        for (int i = 1; i < group_size && j * group_size + i < count; ++i) {
            float v = src[j * group_size + i];
            if (v < g_min) g_min = v;
            if (v > g_max) g_max = v;
        }
        float range = g_max - g_min;
        if (range < 1e-10f) range = 1.0f;
        scales[j] = range / 31.0f;
        float clamped_min = (g_min < 0) ? g_min : 0;
        mins[j] = -clamped_min;
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    block.d = float_to_half(max_scale / 63.0f);
    block.dmin = float_to_half(max_min / 63.0f);

    memset(block.scales, 0, 12);
    for (int j = 0; j < n_groups; ++j) {
        uint8_t ls = std::min(63, std::max(0, static_cast<int>(std::round(scales[j] * inv_scale))));
        uint8_t lm = std::min(63, std::max(0, static_cast<int>(std::round(mins[j] * inv_min))));
        if (j < 4) {
            block.scales[j] = ls;
            block.scales[j + 4] = lm;
        } else {
            block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            block.scales[j - 4] |= ((ls >> 4) << 6);
            block.scales[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint8_t L[256];
    float d_super = half_to_float(block.d);
    float m_super = half_to_float(block.dmin);
    for (int j = 0; j < n_groups; ++j) {
        uint8_t sc, m;
        get_scale_min_k4(j, block.scales, &sc, &m);
        float d_eff = d_super * sc;
        float m_eff = m_super * m;
        if (d_eff < 1e-10f) d_eff = 1.0f;
        for (int i = 0; i < group_size && j * group_size + i < count; ++i) {
            int l = static_cast<int>(std::round((src[j * group_size + i] + m_eff) / d_eff));
            L[j * group_size + i] = static_cast<uint8_t>(std::max(0, std::min(31, l)));
        }
        for (int i = std::max(0, count - j * group_size); i < group_size; ++i) {
            L[j * group_size + i] = 0;
        }
    }

    uint8_t* qh = block.qh;
    memset(qh, 0, 32);
    uint8_t* q = block.qs;
    uint8_t m1 = 1, m2 = 2;
    for (int j = 0; j < 256; j += 64) {
        for (int l = 0; l < 32; ++l) {
            int l1 = L[j + l];
            int l2 = L[j + l + 32];
            if (l1 > 15) { l1 -= 16; qh[l] |= m1; }
            if (l2 > 15) { l2 -= 16; qh[l] |= m2; }
            q[l] = static_cast<uint8_t>(l1 | (l2 << 4));
        }
        q += 32;
        m1 <<= 2; m2 <<= 2;
    }
}

static void quantize_block_q6_K(block_q6_K& block, const float* src, int count) {
    int n_sub = 16;
    int sub_size = 16;
    float max_abs_scale = 0.0f;
    float sub_scales[16];

    for (int j = 0; j < n_sub; ++j) {
        float amax = 0.0f;
        for (int i = 0; i < sub_size && j * sub_size + i < count; ++i) {
            amax = std::max(amax, std::abs(src[j * sub_size + i]));
        }
        sub_scales[j] = (amax < 1e-10f) ? 0.0f : (amax / 32.0f);
        float abs_s = std::abs(sub_scales[j]);
        if (abs_s > max_abs_scale) max_abs_scale = abs_s;
    }

    if (max_abs_scale < 1e-10f) {
        memset(&block, 0, sizeof(block));
        block.d = float_to_half(0.0f);
        return;
    }

    float iscale = -128.0f / max_abs_scale;
    block.d = float_to_half(1.0f / iscale);
    for (int j = 0; j < n_sub; ++j) {
        block.scales[j] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(std::round(iscale * sub_scales[j])))));
    }

    int8_t L[256];
    float d_all = half_to_float(block.d);
    for (int j = 0; j < n_sub; ++j) {
        float d_sub = d_all * block.scales[j];
        if (std::abs(d_sub) < 1e-10f) d_sub = 1.0f;
        for (int i = 0; i < sub_size && j * sub_size + i < count; ++i) {
            int l = static_cast<int>(std::round(src[j * sub_size + i] / d_sub));
            l = std::max(-32, std::min(31, l));
            L[j * sub_size + i] = static_cast<int8_t>(l);
        }
        for (int i = std::max(0, count - j * sub_size); i < sub_size; ++i) {
            L[j * sub_size + i] = 0;
        }
    }

    // Pack: low 4 bits in ql, high 2 bits in qh (GGML batch layout)
    memset(block.ql, 0, 128);
    memset(block.qh, 0, 64);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int v0 = L[n + l] + 32;
            int v1 = L[n + l + 32] + 32;
            int v2 = L[n + l + 64] + 32;
            int v3 = L[n + l + 96] + 32;
            block.ql[n / 2 + l] = (v0 & 0xF) | ((v2 & 0xF) << 4);
            block.ql[n / 2 + l + 32] = (v1 & 0xF) | ((v3 & 0xF) << 4);
        }
        for (int l = 0; l < 32; ++l) {
            int v0 = L[n + l] + 32;
            int v1 = L[n + l + 32] + 32;
            int v2 = L[n + l + 64] + 32;
            int v3 = L[n + l + 96] + 32;
            block.qh[n / 4 + l] = ((v0 >> 4) & 3) | (((v1 >> 4) & 3) << 2) |
                                  (((v2 >> 4) & 3) << 4) | (((v3 >> 4) & 3) << 6);
        }
    }
}

// ── Quantize (public API) ──────────────────────────────────────────────

Expected<void> quantize(Tensor& dst, const Tensor& src, QuantFormat format) {
    auto* src_ptr = src.data<const float>();
    auto* dst_ptr = static_cast<char*>(dst.storage()->data);
    const auto& shape = src.type().shape();

    bool is_kquant = (format == QuantFormat::Q2_K || format == QuantFormat::Q3_K ||
                      format == QuantFormat::Q4_K || format == QuantFormat::Q5_K ||
                      format == QuantFormat::Q6_K);
    int block_size = is_kquant ? 256 : 32;

    // For 2D tensors, quantize each row separately
    if (shape.size() == 2) {
        auto M = shape[0];
        auto K = shape[1];
        size_t blocks_per_row = (static_cast<size_t>(K) + block_size - 1) / block_size;

        switch (format) {
            case QuantFormat::Q8_0: {
                auto* blocks = reinterpret_cast<block_q8_0*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int offset = static_cast<int>(bk * 32);
                        int count = std::min(32, static_cast<int>(K) - offset);
                        quantize_block_q8_0(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q4_0: {
                auto* blocks = reinterpret_cast<block_q4_0*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int offset = static_cast<int>(bk * 32);
                        int count = std::min(32, static_cast<int>(K) - offset);
                        quantize_block_q4_0(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q2_K: {
                auto* blocks = reinterpret_cast<block_q2_K*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        quantize_block_q2_K(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q3_K: {
                auto* blocks = reinterpret_cast<block_q3_K*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        quantize_block_q3_K(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q4_K: {
                auto* blocks = reinterpret_cast<block_q4_K*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        quantize_block_q4_K(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q5_K: {
                auto* blocks = reinterpret_cast<block_q5_K*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        quantize_block_q5_K(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q6_K: {
                auto* blocks = reinterpret_cast<block_q6_K*>(dst_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        quantize_block_q6_K(blocks[row * blocks_per_row + bk], src_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            default:
                return Error{"cpu::quantize: unsupported format"};
        }
    }

    // Flat (1D) quantization
    auto numel = src.type().numel();
    size_t num_blocks = (static_cast<size_t>(numel) + block_size - 1) / block_size;

    switch (format) {
        case QuantFormat::Q8_0: {
            auto* blocks = reinterpret_cast<block_q8_0*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int offset = static_cast<int>(b * 32);
                int count = std::min(32, static_cast<int>(numel) - offset);
                quantize_block_q8_0(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q4_0: {
            auto* blocks = reinterpret_cast<block_q4_0*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int offset = static_cast<int>(b * 32);
                int count = std::min(32, static_cast<int>(numel) - offset);
                quantize_block_q4_0(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q2_K: {
            auto* blocks = reinterpret_cast<block_q2_K*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                quantize_block_q2_K(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q3_K: {
            auto* blocks = reinterpret_cast<block_q3_K*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                quantize_block_q3_K(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q4_K: {
            auto* blocks = reinterpret_cast<block_q4_K*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                quantize_block_q4_K(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q5_K: {
            auto* blocks = reinterpret_cast<block_q5_K*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                quantize_block_q5_K(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q6_K: {
            auto* blocks = reinterpret_cast<block_q6_K*>(dst_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                quantize_block_q6_K(blocks[b], src_ptr + offset, count);
            }
            return {};
        }
        default:
            return Error{"cpu::quantize: unsupported format"};
    }
}

// ── Dequantize ─────────────────────────────────────────────────────────

Expected<void> dequantize(Tensor& dst, const Tensor& src) {
    auto* dst_ptr = dst.data<float>();
    auto* src_ptr = static_cast<const char*>(src.storage()->data);
    const auto& shape = src.type().shape();
    QuantFormat format = src.storage()->quant.format;

    auto dequantize_block_q8_0 = [&](const block_q8_0& block, float* out, int count) {
        float d = half_to_float(block.d);
        for (int i = 0; i < count; ++i) out[i] = static_cast<float>(block.qs[i]) * d;
    };

    auto dequantize_block_q4_0 = [&](const block_q4_0& block, float* out, int count) {
        float d = half_to_float(block.d);
        for (int i = 0; i < 16 && i * 2 < count; ++i) {
            int idx_lo = i * 2;
            int idx_hi = i * 2 + 1;
            if (idx_lo < count) {
                int8_t q = static_cast<int8_t>(block.qs[i] & 0xF) - 8;
                out[idx_lo] = static_cast<float>(q) * d;
            }
            if (idx_hi < count) {
                int8_t q = static_cast<int8_t>(block.qs[i] >> 4) - 8;
                out[idx_hi] = static_cast<float>(q) * d;
            }
        }
    };

    auto dequantize_block_q2_K = [&](const block_q2_K& block, float* out, int count) {
        float d = half_to_float(block.d);
        float min = half_to_float(block.dmin);
        const uint8_t* q = block.qs;
        int is = 0;
        int written = 0;
        for (int n = 0; n < 256 && written < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = block.scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16 && written < count; ++l) {
                    out[written++] = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
                }
                sc = block.scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16 && written < count; ++l) {
                    out[written++] = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
                }
                shift += 2;
            }
            q += 32;
        }
    };

    auto dequantize_block_q3_K = [&](const block_q3_K& block, float* out, int count) {
        float d_all = half_to_float(block.d);
        const uint8_t* q = block.qs;
        const uint8_t* hm = block.hmask;
        const int8_t* scales_ptr = nullptr;
        uint32_t aux[4];
        uint32_t kmask1 = 0x03030303;
        uint32_t kmask2 = 0x0f0f0f0f;
        memcpy(aux, block.scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        scales_ptr = reinterpret_cast<const int8_t*>(aux);

        uint8_t m = 1;
        int is = 0;
        int written = 0;
        for (int n = 0; n < 256 && written < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales_ptr[is++] - 32);
                for (int l = 0; l < 16 && written < count; ++l) {
                    int8_t qv = static_cast<int8_t>((q[l] >> shift) & 3);
                    out[written++] = dl * (qv - ((hm[l] & m) ? 0 : 4));
                }
                dl = d_all * static_cast<float>(scales_ptr[is++] - 32);
                for (int l = 0; l < 16 && written < count; ++l) {
                    int8_t qv = static_cast<int8_t>((q[l + 16] >> shift) & 3);
                    out[written++] = dl * (qv - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    };

    auto dequantize_block_q4_K = [&](const block_q4_K& block, float* out, int count) {
        const uint8_t* q = block.qs;
        float d = half_to_float(block.d);
        float min = half_to_float(block.dmin);
        int is = 0;
        int written = 0;
        for (int j = 0; j < 256 && written < count; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            float d1 = d * sc;
            float m1 = min * m;
            get_scale_min_k4(is + 1, block.scales, &sc, &m);
            float d2 = d * sc;
            float m2 = min * m;
            for (int l = 0; l < 32 && written < count; ++l) {
                out[written++] = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32 && written < count; ++l) {
                out[written++] = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    };

    auto dequantize_block_q5_K = [&](const block_q5_K& block, float* out, int count) {
        const uint8_t* ql = block.qs;
        const uint8_t* qh = block.qh;
        float d = half_to_float(block.d);
        float min = half_to_float(block.dmin);
        int is = 0;
        int written = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256 && written < count; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, block.scales, &sc, &m);
            float d1 = d * sc;
            float m1 = min * m;
            get_scale_min_k4(is + 1, block.scales, &sc, &m);
            float d2 = d * sc;
            float m2 = min * m;
            for (int l = 0; l < 32 && written < count; ++l) {
                out[written++] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32 && written < count; ++l) {
                out[written++] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            ql += 32;
            u1 <<= 2; u2 <<= 2;
            is += 2;
        }
    };

    auto dequantize_block_q6_K = [&](const block_q6_K& block, float* out, int count) {
        float d = half_to_float(block.d);
        const uint8_t* ql = block.ql;
        const uint8_t* qh = block.qh;
        const int8_t* sc = block.scales;
        int written = 0;
        for (int n = 0; n < 256 && written < count; n += 128) {
            for (int l = 0; l < 32 && written < count; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                if (written < count) out[written] = d * sc[is] * q1;
                if (written + 32 < count) out[written + 32] = d * sc[is + 2] * q2;
                if (written + 64 < count) out[written + 64] = d * sc[is + 4] * q3;
                if (written + 96 < count) out[written + 96] = d * sc[is + 6] * q4;
                ++written;
            }
            written += 3 * 32;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    };

    bool is_kquant = (format == QuantFormat::Q2_K || format == QuantFormat::Q3_K ||
                      format == QuantFormat::Q4_K || format == QuantFormat::Q5_K ||
                      format == QuantFormat::Q6_K);
    int block_size = is_kquant ? 256 : 32;

    if (shape.size() == 2) {
        auto M = shape[0];
        auto K = shape[1];
        size_t blocks_per_row = (static_cast<size_t>(K) + block_size - 1) / block_size;

        switch (format) {
            case QuantFormat::Q8_0: {
                auto* blocks = reinterpret_cast<const block_q8_0*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int offset = static_cast<int>(bk * 32);
                        int count = std::min(32, static_cast<int>(K) - offset);
                        dequantize_block_q8_0(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q4_0: {
                auto* blocks = reinterpret_cast<const block_q4_0*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int offset = static_cast<int>(bk * 32);
                        int count = std::min(32, static_cast<int>(K) - offset);
                        dequantize_block_q4_0(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q2_K: {
                auto* blocks = reinterpret_cast<const block_q2_K*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        dequantize_block_q2_K(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q3_K: {
                auto* blocks = reinterpret_cast<const block_q3_K*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        dequantize_block_q3_K(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q4_K: {
                auto* blocks = reinterpret_cast<const block_q4_K*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        dequantize_block_q4_K(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q5_K: {
                auto* blocks = reinterpret_cast<const block_q5_K*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        dequantize_block_q5_K(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            case QuantFormat::Q6_K: {
                auto* blocks = reinterpret_cast<const block_q6_K*>(src_ptr);
                for (int64_t row = 0; row < M; ++row) {
                    for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                        int64_t offset = static_cast<int64_t>(bk) * 256;
                        int count = static_cast<int>(std::min(static_cast<int64_t>(256), K - offset));
                        dequantize_block_q6_K(blocks[row * blocks_per_row + bk], dst_ptr + row * K + offset, count);
                    }
                }
                return {};
            }
            default:
                return Error{"cpu::dequantize: unsupported format"};
        }
    }

    // Flat (1D) dequantization
    auto numel = src.type().numel();
    size_t num_blocks = (static_cast<size_t>(numel) + block_size - 1) / block_size;

    switch (format) {
        case QuantFormat::Q8_0: {
            auto* blocks = reinterpret_cast<const block_q8_0*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int offset = static_cast<int>(b * 32);
                int count = std::min(32, static_cast<int>(numel) - offset);
                dequantize_block_q8_0(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q4_0: {
            auto* blocks = reinterpret_cast<const block_q4_0*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int offset = static_cast<int>(b * 32);
                int count = std::min(32, static_cast<int>(numel) - offset);
                dequantize_block_q4_0(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q2_K: {
            auto* blocks = reinterpret_cast<const block_q2_K*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                dequantize_block_q2_K(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q3_K: {
            auto* blocks = reinterpret_cast<const block_q3_K*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                dequantize_block_q3_K(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q4_K: {
            auto* blocks = reinterpret_cast<const block_q4_K*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                dequantize_block_q4_K(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q5_K: {
            auto* blocks = reinterpret_cast<const block_q5_K*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                dequantize_block_q5_K(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        case QuantFormat::Q6_K: {
            auto* blocks = reinterpret_cast<const block_q6_K*>(src_ptr);
            for (size_t b = 0; b < num_blocks; ++b) {
                int64_t offset = static_cast<int64_t>(b) * 256;
                int count = static_cast<int>(std::min(static_cast<int64_t>(256), static_cast<int64_t>(numel) - offset));
                dequantize_block_q6_K(blocks[b], dst_ptr + offset, count);
            }
            return {};
        }
        default:
            return Error{"cpu::dequantize: unsupported format"};
    }
}

// ── Q4_0 matmul ────────────────────────────────────────────────────────

Expected<void> matmul_q4(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"cpu::matmul_q4: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"cpu::matmul_q4: inner dimension mismatch"};
    }

    auto fn = KernelRegistry::instance().dispatch("matmul_q4_0");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {a, b};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q4_0*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d = half_to_float(block.d);
                for (int kk = 0; kk < 16; ++kk) {
                    int k_base = static_cast<int>(bk * 32 + kk * 2);

                    int k_lo = k_base;
                    int k_hi = k_base + 1;

                    float d_lo = static_cast<float>(static_cast<int8_t>(block.qs[kk] & 0xF) - 8) * d;
                    float d_hi = static_cast<float>(static_cast<int8_t>(block.qs[kk] >> 4) - 8) * d;

                    if (k_lo < K) sum += d_lo * b_ptr[static_cast<int64_t>(k_lo) * N + j];
                    if (k_hi < K) sum += d_hi * b_ptr[static_cast<int64_t>(k_hi) * N + j];
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }

    return {};
}

// ── Q4_K matmul ────────────────────────────────────────────────────────

Expected<void> matmul_q4_K(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"cpu::matmul_q4_K: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"cpu::matmul_q4_K: inner dimension mismatch"};
    }

    auto fn = KernelRegistry::instance().dispatch("matmul_q4_K");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {a, b};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q4_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    auto get_scale_min = [](int j, const uint8_t* sc_ptr, uint8_t* d_out, uint8_t* m_out) {
        if (j < 4) {
            *d_out = sc_ptr[j] & 63;
            *m_out = sc_ptr[j + 4] & 63;
        } else {
            *d_out = (sc_ptr[j + 4] & 0xF) | ((sc_ptr[j - 4] >> 6) << 4);
            *m_out = (sc_ptr[j + 4] >> 4) | ((sc_ptr[j] >> 6) << 4);
        }
    };

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d_super = half_to_float(block.d);
                float m_super = half_to_float(block.dmin);
                const uint8_t* q = block.qs;
                int is = 0;
                for (int g = 0; g < 256; g += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is, block.scales, &sc, &m);
                    float d1 = d_super * sc;
                    float m1 = m_super * m;
                    get_scale_min_k4(is + 1, block.scales, &sc, &m);
                    float d2 = d_super * sc;
                    float m2 = m_super * m;
                    for (int l = 0; l < 32; ++l) {
                        int k_idx = static_cast<int>(bk * 256 + g + l);
                        if (k_idx < K) {
                            float val = d1 * (q[l] & 0xF) - m1;
                            sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                        }
                    }
                    for (int l = 0; l < 32; ++l) {
                        int k_idx = static_cast<int>(bk * 256 + g + 32 + l);
                        if (k_idx < K) {
                            float val = d2 * (q[l] >> 4) - m2;
                            sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
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

// ── Q6_K matmul ────────────────────────────────────────────────────────

Expected<void> matmul_q6_K(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"cpu::matmul_q6_K: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"cpu::matmul_q6_K: inner dimension mismatch"};
    }

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q6_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d = half_to_float(block.d);
                const uint8_t* ql = block.ql;
                const uint8_t* qh = block.qh;
                const int8_t* sc = block.scales;
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; ++l) {
                        int is = l / 16;
                        int8_t q1 = static_cast<int8_t>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                        int k0 = static_cast<int>(bk * 256 + n + l);
                        int k1 = static_cast<int>(bk * 256 + n + l + 32);
                        int k2 = static_cast<int>(bk * 256 + n + l + 64);
                        int k3 = static_cast<int>(bk * 256 + n + l + 96);

                        if (k0 < K) sum += d * sc[is]     * q1 * b_ptr[static_cast<int64_t>(k0) * N + j];
                        if (k1 < K) sum += d * sc[is + 2] * q2 * b_ptr[static_cast<int64_t>(k1) * N + j];
                        if (k2 < K) sum += d * sc[is + 4] * q3 * b_ptr[static_cast<int64_t>(k2) * N + j];
                        if (k3 < K) sum += d * sc[is + 6] * q4 * b_ptr[static_cast<int64_t>(k3) * N + j];
                    }
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }
    return {};
}

// ── Q2_K matmul ────────────────────────────────────────────────────────

Expected<void> matmul_q2_K(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"cpu::matmul_q2_K: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"cpu::matmul_q2_K: inner dimension mismatch"};
    }

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q2_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d = half_to_float(block.d);
                float min = half_to_float(block.dmin);
                const uint8_t* q = block.qs;
                int is = 0;
                for (int n = 0; n < 256; n += 128) {
                    int shift = 0;
                    for (int s = 0; s < 4; ++s) {
                        uint8_t sc = block.scales[is++];
                        float dl = d * (sc & 0xF);
                        float ml = min * (sc >> 4);
                        for (int l = 0; l < 16; ++l) {
                            int k_idx = static_cast<int>(bk * 256 + n + s * 32 + l);
                            if (k_idx < K) {
                                float val = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
                                sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                            }
                        }
                        sc = block.scales[is++];
                        dl = d * (sc & 0xF);
                        ml = min * (sc >> 4);
                        for (int l = 0; l < 16; ++l) {
                            int k_idx = static_cast<int>(bk * 256 + n + s * 32 + 16 + l);
                            if (k_idx < K) {
                                float val = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
                                sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                            }
                        }
                        shift += 2;
                    }
                    q += 32;
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }
    return {};
}

// ── Q3_K matmul ────────────────────────────────────────────────────────

Expected<void> matmul_q3_K(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"cpu::matmul_q3_K: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"cpu::matmul_q3_K: inner dimension mismatch"};
    }

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q3_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d_all = half_to_float(block.d);
                const uint8_t* q = block.qs;
                const uint8_t* hm = block.hmask;

                // Unpack scales (same as dequantize)
                uint32_t aux[4];
                uint32_t kmask1 = 0x03030303;
                uint32_t kmask2 = 0x0f0f0f0f;
                memcpy(aux, block.scales, 12);
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                const int8_t* scales_ptr = reinterpret_cast<const int8_t*>(aux);

                uint8_t m = 1;
                int is = 0;
                for (int n = 0; n < 256; n += 128) {
                    int shift = 0;
                    for (int s = 0; s < 4; ++s) {
                        float dl = d_all * static_cast<float>(scales_ptr[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            int k_idx = static_cast<int>(bk * 256 + n + s * 32 + l);
                            if (k_idx < K) {
                                int8_t qv = static_cast<int8_t>((q[l] >> shift) & 3);
                                float val = dl * (qv - ((hm[l] & m) ? 0 : 4));
                                sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                            }
                        }
                        dl = d_all * static_cast<float>(scales_ptr[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            int k_idx = static_cast<int>(bk * 256 + n + s * 32 + 16 + l);
                            if (k_idx < K) {
                                int8_t qv = static_cast<int8_t>((q[l + 16] >> shift) & 3);
                                float val = dl * (qv - ((hm[l + 16] & m) ? 0 : 4));
                                sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                            }
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }
    return {};
}

// ── Q5_K matmul ────────────────────────────────────────────────────────

Expected<void> matmul_q5_K(Tensor& out, const Tensor& a, const Tensor& b) {
    if (a.type().shape().size() != 2 || b.type().shape().size() != 2) {
        return Error{"cpu::matmul_q5_K: inputs must be 2D"};
    }
    if (a.type().shape()[1] != b.type().shape()[0]) {
        return Error{"cpu::matmul_q5_K: inner dimension mismatch"};
    }

    auto fn = KernelRegistry::instance().dispatch("matmul_q5_K");
    if (fn) {
        Tensor outputs[] = {out};
        Tensor inputs[] = {a, b};
        KernelContext ctx{.outputs = outputs, .inputs = inputs};
        return fn(ctx);
    }

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q5_K*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 255) / 256;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t bk = 0; bk < blocks_per_row; ++bk) {
                const auto& block = blocks[i * blocks_per_row + bk];
                float d_super = half_to_float(block.d);
                float m_super = half_to_float(block.dmin);
                const uint8_t* ql = block.qs;
                const uint8_t* qh = block.qh;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int g = 0; g < 256; g += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is, block.scales, &sc, &m);
                    float d1 = d_super * sc;
                    float m1 = m_super * m;
                    get_scale_min_k4(is + 1, block.scales, &sc, &m);
                    float d2 = d_super * sc;
                    float m2 = m_super * m;
                    for (int l = 0; l < 32; ++l) {
                        int k_idx = static_cast<int>(bk * 256 + g + l);
                        if (k_idx < K) {
                            float val = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                            sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                        }
                    }
                    for (int l = 0; l < 32; ++l) {
                        int k_idx = static_cast<int>(bk * 256 + g + 32 + l);
                        if (k_idx < K) {
                            float val = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                            sum += val * b_ptr[static_cast<int64_t>(k_idx) * N + j];
                        }
                    }
                    ql += 32;
                    u1 <<= 2; u2 <<= 2;
                    is += 2;
                }
            }
            o_ptr[i * N + j] = sum;
        }
    }
    return {};
}

} // namespace axon::cpu
