/*
 * mf_dispatch.h — runtime CPU dispatch for the multifloats DD-SIMD kernels.
 *
 * The library is built at a portable baseline -march (e.g. sandybridge) so a
 * single shipped binary runs on pre-Haswell CPUs. The AVX2/FMA double-double
 * SIMD kernels are still COMPILED IN: every SIMD function carries
 * MF_SIMD_TARGET — an explicit target("avx2,fma") attribute — and the intrinsic
 * primitive headers (mf_simd_fast.h / mf_simd_exact.h / mf_kernels.h) wrap their
 * bodies in `#pragma GCC target("avx2,fma")`, so the AVX2/FMA intrinsics compile
 * even when the library's baseline -march is a pre-Haswell ISA.
 *
 * The SIMD path is ENTERED only behind mf_have_avx2_fma(): on Haswell-or-newer
 * the fast DD-SIMD kernels run; on Sandybridge/Ivy Bridge the CPU takes the
 * scalar fallback and never executes an AVX2/FMA instruction. One binary is
 * therefore both Sandybridge-safe and Haswell-fast.
 *
 * Non-x86 builds: MF_SIMD_TARGET expands to nothing and mf_have_avx2_fma() is a
 * compile-time false, so with MBLAS_SIMD_DD undefined the SIMD bodies compile
 * out and only the scalar path remains.
 */
#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#  define MF_SIMD_TARGET __attribute__((target("avx2,fma")))
#  ifdef __cplusplus
extern "C" {
#  endif
/* Cached CPUID probe: nonzero iff the running CPU has both AVX2 and FMA. */
int mf_have_avx2_fma(void);
#  ifdef __cplusplus
}
#  endif
#else
#  define MF_SIMD_TARGET
#  ifdef __cplusplus
static inline int mf_have_avx2_fma(void) { return 0; }
#  else
static inline int mf_have_avx2_fma(void) { return 0; }
#  endif
#endif
