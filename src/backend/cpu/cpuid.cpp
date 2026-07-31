#include "axon/backend/cpuid.h"

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

namespace axon::cpu {

static void run_cpuid(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    __cpuidex(regs, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

static uint64_t run_xgetbv(uint32_t xcr) {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#else
    return 0;
#endif
}

CpuFeatures detect_cpu_features() {
    CpuFeatures feat;

    int regs[4] = {0};
    run_cpuid(1, 0, regs);

    bool osxsave = (regs[2] & (1 << 27)) != 0;
    bool avx = (regs[2] & (1 << 28)) != 0;
    bool fma = (regs[2] & (1 << 12)) != 0;

    bool ymm_supported = false;
    if (osxsave) {
        uint64_t xcr0 = run_xgetbv(0);
        // Bit 1 (XMM) and Bit 2 (YMM) must be enabled by OS
        ymm_supported = (xcr0 & 6) == 6;
    }

    if (avx && ymm_supported) {
        feat.avx = true;
        feat.fma3 = fma;

        int leaf7_regs[4] = {0};
        run_cpuid(7, 0, leaf7_regs);

        feat.avx2 = (leaf7_regs[1] & (1 << 5)) != 0;
        feat.avx512f = (leaf7_regs[1] & (1 << 16)) != 0;
        feat.avx512vnni = (leaf7_regs[2] & (1 << 11)) != 0;
    }

    return feat;
}

static CpuFeatures g_cpu_features = detect_cpu_features();

bool has_avx2() {
    return g_cpu_features.avx2 && g_cpu_features.fma3;
}

ISA get_best_isa() {
    if (has_avx2()) {
        return ISA::AVX2;
    }
    return ISA::Scalar;
}

} // namespace axon::cpu
