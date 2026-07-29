#include "axon/backend/cpu_backend.h"
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
    auto* rm = running_mean.data<float>();
    auto* rv = running_var.data<float>();
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

        // Update running stats
        for (int64_t c = 0; c < C; ++c) {
            rm[c] = momentum * rm[c] + (1.0f - momentum) * mean[c];
            rv[c] = momentum * rv[c] + (1.0f - momentum) * var[c];
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

// ── Quantized size ─────────────────────────────────────────────────────

size_t quantized_size(size_t num_elements, QuantFormat format) {
    size_t num_blocks = (num_elements + 31) / 32;
    switch (format) {
        case QuantFormat::Q8_0: return num_blocks * sizeof(block_q8_0);
        case QuantFormat::Q4_0: return num_blocks * sizeof(block_q4_0);
        default: return 0;
    }
}

size_t quantized_size_2d(int64_t M, int64_t K, QuantFormat format) {
    size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;
    size_t total_blocks = static_cast<size_t>(M) * blocks_per_row;
    switch (format) {
        case QuantFormat::Q8_0: return total_blocks * sizeof(block_q8_0);
        case QuantFormat::Q4_0: return total_blocks * sizeof(block_q4_0);
        default: return 0;
    }
}

// ── Quantize ───────────────────────────────────────────────────────────

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

Expected<void> quantize(Tensor& dst, const Tensor& src, QuantFormat format) {
    auto* src_ptr = src.data<const float>();
    auto* dst_ptr = static_cast<char*>(dst.storage()->data);
    const auto& shape = src.type().shape();

    // For 2D tensors, quantize each row separately (row-major block layout for matmul)
    if (shape.size() == 2) {
        auto M = shape[0];
        auto K = shape[1];
        size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;

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
            default:
                return Error{"cpu::quantize: unsupported format"};
        }
    }

    // Flat (1D) quantization
    auto numel = src.type().numel();
    size_t num_blocks = (static_cast<size_t>(numel) + 31) / 32;

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

    if (shape.size() == 2) {
        auto M = shape[0];
        auto K = shape[1];
        size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;

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
            default:
                return Error{"cpu::dequantize: unsupported format"};
        }
    }

    // Flat (1D) dequantization
    auto numel = src.type().numel();
    size_t num_blocks = (static_cast<size_t>(numel) + 31) / 32;

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

    auto M = a.type().shape()[0];
    auto K = a.type().shape()[1];
    auto N = b.type().shape()[1];
    auto* b_ptr = b.data<const float>();
    auto* o_ptr = out.data<float>();
    auto* a_data = static_cast<const char*>(a.storage()->data);
    auto* blocks = reinterpret_cast<const block_q4_0*>(a_data);
    size_t blocks_per_row = (static_cast<size_t>(K) + 31) / 32;

    // Dequantize blocks on-the-fly for each output element
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

} // namespace axon::cpu
