/*
 * mf_cpu.cpp — runtime CPU feature probe backing mf_dispatch.h.
 *
 * Compiled at the library's baseline -march (NOT with -mavx2/-mfma): this
 * function must be safe to call on the oldest supported CPU, since it is what
 * decides whether the AVX2/FMA DD-SIMD kernels may run at all.
 */
#include "mf_dispatch.h"

#if defined(__x86_64__) || defined(_M_X64)

extern "C" int mf_have_avx2_fma(void)
{
    /* GCC/Clang emit the __builtin_cpu_init() constructor automatically, so the
     * feature bits are populated before main(). Cache the AND of AVX2 and FMA
     * in a function-local static (thread-safe init in C++) so the hot BLAS
     * dispatch branch is a single predictable load after the first call. */
    static const int has = (__builtin_cpu_supports("avx2")
                            && __builtin_cpu_supports("fma")) ? 1 : 0;
    return has;
}

#endif
