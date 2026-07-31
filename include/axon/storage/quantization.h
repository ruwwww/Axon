#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "axon/core/types.h"

namespace axon {

struct QuantizationDescriptor {
    QuantFormat format = QuantFormat::None;
    size_t block_size = 0;
};

inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exp == 0) {
        if (mantissa == 0) {
            uint32_t u = sign << 31;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exp--;
        }
        exp++;
        mantissa &= ~0x0400;
    } else if (exp == 31) {
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

inline uint16_t float_to_half(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));

    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = u & 0x007FFFFF;

    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};

struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};

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

} // namespace axon
