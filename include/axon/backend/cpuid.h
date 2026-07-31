#pragma once

#include <cstdint>

namespace axon::cpu {

enum class ISA : uint8_t {
    Scalar = 0,
    AVX2 = 1,
    AVX512 = 2,
    NEON = 3
};

struct CpuFeatures {
    bool avx = false;
    bool avx2 = false;
    bool fma3 = false;
    bool avx512f = false;
    bool avx512vnni = false;
    bool avx_vnni = false;
};

CpuFeatures detect_cpu_features();
bool has_avx2();
bool has_avx_vnni();
ISA get_best_isa();

} // namespace axon::cpu
