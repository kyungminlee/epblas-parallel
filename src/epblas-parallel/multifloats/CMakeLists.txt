# Hand-written parallel BLAS — multifloats target (double-double).
#
# Type: upstream `float64x2` (2 IEEE doubles, ~32-digit precision).
# Each DD op is an error-free transform: ~8 scalar flops per add,
# ~12 per mul. Arithmetic-bound; OpenMP scales nearly linearly.
#
# Uses the multifloats C API (adddd/muldd/fmadd/cmuldd/etc.) from
# <multifloats/float64x2.h>. The multifloats library is provided by
# the upstream FetchContent target `multifloats`.

set(_kernels
    wsyrk.cpp
    wherk.cpp
    msymm.cpp
    wsymm.cpp
    whemm.cpp
    msyr2k.cpp
    wsyr2k.cpp
    wher2k.cpp
    mgemv.cpp
    wgemv.cpp
    mger.cpp
    wgeru.cpp
    wgerc.cpp
    msymv.cpp
    whemv.cpp
    mtrsv.cpp
    wtrsv.cpp
    mtrmv.cpp
    wtrmv.cpp
    msyr.cpp
    wher.cpp
    msyr2.cpp
    wher2.cpp
    # L1 BLAS — subroutine-style (function-return L1 routines deferred)
    mscal.cpp
    wscal.cpp
    wmscal.cpp
    maxpy.cpp
    waxpy.cpp
    mcopy.cpp
    wcopy.cpp
    mswap.cpp
    wswap.cpp
    mrot.cpp
    wmrot.cpp
    # L1 function-return overlays — DD struct returns via SysV register
    # pair (16B) or sret (32B); gfortran convention matches since the
    # multifloats types are bind(c).
    mdot.cpp
    masum.cpp
    mnrm2.cpp
    mwasum.cpp
    mwnrm2.cpp
    wdotu.cpp
    wdotc.cpp
    # L2 banded — general
    mgbmv.cpp
    wgbmv.cpp
    # L2 banded — symmetric / Hermitian
    msbmv.cpp
    whbmv.cpp
    # L2 packed — symmetric / Hermitian mat-vec
    mspmv.cpp
    whpmv.cpp
    # L2 banded — triangular mat-vec / solve
    mtbmv.cpp
    wtbmv.cpp
    mtbsv.cpp
    wtbsv.cpp
    # L2 packed — triangular mat-vec / solve
    mtpmv.cpp
    wtpmv.cpp
    mtpsv.cpp
    wtpsv.cpp
    # L2 packed — sym/Herm rank-1 + rank-2
    mspr.cpp
    whpr.cpp
    mspr2.cpp
    whpr2.cpp
    # L1 misc — cabs1, iamax, Givens generators
    mcabs1.cpp
    imamax.cpp
    iwamax.cpp
    mrotg.cpp
    mrotm.cpp
    mrotmg.cpp
    wrotg.cpp
)

# L3 routines split kind10/kind16-style into a public threading driver
# (*_parallel.cpp) + an internal single-thread worker (*_serial.cpp) that
# owns ALL the numerics — packers, micro-kernels (scalar AND the bespoke
# AVX2 SIMD double-double kernels), block policy — shared through an
# include-only *_kernel.h. The serial worker is what the other L3 routines
# call for trailing updates inside their own parallel region (replacing the
# old reliance on OpenMP's default-off nested parallelism collapsing the
# inner team to one). Only the bare routine name (mgemm) goes into
# _routines — the split filenames must NOT be stripped to *_parallel /
# *_serial pseudo-routines.
set(_split_sources
    mgemm_parallel.cpp
    mgemm_serial.cpp
    wgemm_parallel.cpp
    wgemm_serial.cpp
    mgemmtr_parallel.cpp
    mgemmtr_serial.cpp
    wgemmtr_parallel.cpp
    wgemmtr_serial.cpp
    mtrsm_parallel.cpp
    mtrsm_serial.cpp
    wtrsm_parallel.cpp
    wtrsm_serial.cpp
    mtrmm_parallel.cpp
    mtrmm_serial.cpp
    wtrmm_parallel.cpp
    wtrmm_serial.cpp
    msyrk_parallel.cpp
    msyrk_serial.cpp
)

set(_routines "")
foreach(_k ${_kernels})
    string(REGEX REPLACE "\\.cpp$" "" _r "${_k}")
    list(APPEND _routines "${_r}")
endforeach()
list(APPEND _routines mgemm wgemm mgemmtr wgemmtr mtrsm wtrsm mtrmm wtrmm msyrk)
set(EPBLAS_PARALLEL_OVERLAY_ROUTINES "${_routines}" PARENT_SCOPE)

add_library(${LIB_PREFIX}blas_parallel STATIC ${_kernels} ${_split_sources})
add_library(epblas-parallel::${LIB_PREFIX}blas_parallel ALIAS ${LIB_PREFIX}blas_parallel)

# C++17 for std::aligned_alloc + multifloats's header requirements.
target_compile_features(${LIB_PREFIX}blas_parallel PRIVATE cxx_std_17)
target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
    -O3
    # multifloats forbids -ffast-math (breaks IEEE FMA semantics that
    # the error-free transforms rely on). Use -ffp-contract=on, not
    # =fast — fast lets the compiler fold across statements.
    -ffp-contract=on
)
if(NOT "${EPBLAS_PARALLEL_MARCH}" STREQUAL "")
    target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
        -march=${EPBLAS_PARALLEL_MARCH})
endif()

# multifloats C library provides adddd / muldd / fmadd / cmuldd / ...
target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC multifloats)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC OpenMP::OpenMP_CXX)
elseif(OpenMP_C_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC OpenMP::OpenMP_C)
endif()

# AVX2 SIMD double-double kernel for mgemm. 4-wide SoA — see
# mgemm_simd_kernel.h. Default ON for x86_64; the cmake check below
# disables it if -mavx2 isn't accepted by the compiler. Bench result
# on Raptor Lake: 3.8× over the scalar outer-product baseline single-
# threaded, 38× over migrated at OMP=4 s=1024.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2 -mfma" _HAS_AVX2_FMA_FLAGS)
option(MBLAS_SIMD_DD
    "multifloats: AVX2 SIMD kernel for mgemm (4-wide SoA double-double)"
    ${_HAS_AVX2_FMA_FLAGS})
if(MBLAS_SIMD_DD)
    target_compile_definitions(${LIB_PREFIX}blas_parallel PRIVATE
        MBLAS_SIMD_DD=1)
    target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
        -mavx2 -mfma)
    # MR — register-tile rows for the SIMD micro-kernel.
    # Empirical sweep on Raptor Lake (i3-1315U) at OMP=1 s=1024:
    #   MR=1: 1.92 GFLOP/s            (baseline, 1 chain)
    #   MR=2: 3.49     (+82%, ILP win — two independent dd_mul chains)
    #   MR=3: 3.74     (+7%, sweet spot at s=1024)
    #   MR=4: 3.78     (+1%, register pressure cancels parallelism gain)
    # Re-measured at s=128/256 with 80 iters pinned to a P-core:
    #   MR=4 is +5% over MR=3 with stdev 0.2-0.4% — clean structural win.
    #   At s=64 the two are statistically indistinguishable (-1.2% mean,
    #   3.1% stdev — within noise). Pick MR=4 as the new default.
    set(MBLAS_SIMD_MR 4 CACHE STRING
        "multifloats: SIMD micro-kernel register-tile rows MR (1..4)")
    # NR_PAN: # of ymm-wide (4-col) panels per micro-kernel call.
    # 1 → NR=4 (current default). 2 → NR=8 (forces MR≤2 due to regs).
    set(MBLAS_SIMD_NR_PAN 1 CACHE STRING
        "multifloats: number of 4-lane B-panels per micro-kernel (1..2)")
    target_compile_definitions(${LIB_PREFIX}blas_parallel PRIVATE
        MGEMM_SIMD_MR=${MBLAS_SIMD_MR}
        MGEMM_SIMD_NR_PAN=${MBLAS_SIMD_NR_PAN})

    # Same SIMD treatment for wgemm (complex DD). Default ON; flips off
    # together with MBLAS_SIMD_DD if AVX2+FMA isn't supported.
    option(WBLAS_SIMD_DD
        "multifloats: AVX2 SIMD kernel for wgemm (complex DD)"
        ${_HAS_AVX2_FMA_FLAGS})
    if(WBLAS_SIMD_DD)
        target_compile_definitions(${LIB_PREFIX}blas_parallel PRIVATE
            WBLAS_SIMD_DD=1)
        # Per-cell complex SIMD already has 4-way intra-cell ILP — start
        # at MR=1, NR_PAN=1 and sweep.
        set(WBLAS_SIMD_MR 1 CACHE STRING
            "multifloats: wgemm SIMD micro-kernel rows MR (1..2)")
        set(WBLAS_SIMD_NR_PAN 1 CACHE STRING
            "multifloats: wgemm SIMD 4-lane panels per call (1..2)")
        target_compile_definitions(${LIB_PREFIX}blas_parallel PRIVATE
            WGEMM_SIMD_MR=${WBLAS_SIMD_MR}
            WGEMM_SIMD_NR_PAN=${WBLAS_SIMD_NR_PAN})
    endif()
endif()
