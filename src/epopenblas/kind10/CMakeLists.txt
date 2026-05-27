# epopenblas — kind10 (REAL(KIND=10) / 80-bit long double).
#
# OpenBLAS D and Z source ports, retyped to long double / _Complex long double.
# Builds a STATIC archive `${LIB_PREFIX}blas_epopenblas` consumed only by
# tests/blas_epopenblas/. NOT wired into the public ${LIB_PREFIX}blas composite.
#
# Adding a routine:
#   1. Drop the ported source into this directory, e.g. eaxpy.c
#   2. Append the filename to _kernels below.
#   3. The matching tests/blas_epopenblas/.../target_kind10/ shim picks it up
#      automatically — see CHECKLIST.md for the per-routine status table.

set(_kernels
    # ── L1 real ──
    eaxpy.c
    ecopy.c
    eswap.c
    escal.c
    edot.c
    enrm2.c
    easum.c
    ieamax.c
    erot.c
    erotg.c
    erotm.c
    erotmg.c
    # ── L1 complex ──
    yaxpy.c
    ycopy.c
    yswap.c
    yscal.c
    yescal.c
    ydotu.c
    ydotc.c
    eynrm2.c
    eyasum.c
    yerot.c
    yrotg.c
    iyamax.c
    ecabs1.c
    # ── L2 real ──
    egemv.c
    eger.c
    esymv.c
    esyr.c
    esyr2.c
    espr.c
    espr2.c
    espmv.c
    esbmv.c
    egbmv.c
    etrmv.c
    etrsv.c
    etbmv.c
    etbsv.c
    etpmv.c
    etpsv.c
    # ── L2 complex (general / triangular / band / packed) ──
    ygemv.c
    ygeru.c
    ygerc.c
    ygbmv.c
    ytrmv.c
    ytrsv.c
    ytbmv.c
    ytbsv.c
    ytpmv.c
    ytpsv.c
    # ── L2 complex Hermitian ──
    yhemv.c
    yhbmv.c
    yher.c
    yher2.c
    yhpmv.c
    yhpr.c
    yhpr2.c
    # ── L3 real ──
    egemm.c
    esymm.c
    esyrk.c
    esyr2k.c
    etrmm.c
    etrsm.c
    egemmtr.c
    # ── L3 complex ──
    ygemm.c
    ysymm.c
    ysyrk.c
    ysyr2k.c
    ytrmm.c
    ytrsm.c
    ygemmtr.c
    yhemm.c
    yherk.c
    yher2k.c
)

# Internal helper sources — shared L3 kernel + packers (ports of OpenBLAS
# kernel/generic/{gemmkernel_2x2, gemm_{ncopy,tcopy}_2, gemm_beta}.c for
# the real path and the matching zgemm_*.c for the complex path). These
# contribute symbols to the archive but are not Fortran-callable, so
# they're listed separately from `_kernels` and NOT included in
# `EPOPENBLAS_OVERLAY_ROUTINES`.
set(_helpers
    ../common/eblas_l3_real.c
    ../common/eblas_l3_complex.c
)

# Publish the overlay-covered routine set so the BLAS section in
# cmake/CMakeLists.txt knows which symbols to objcopy-rename in the
# matching `${LIB_PREFIX}blas_epopenblas_migrated` archive.
set(_routines "")
foreach(_k ${_kernels})
    string(REGEX REPLACE "\\.c$" "" _r "${_k}")
    list(APPEND _routines "${_r}")
endforeach()
set(EPOPENBLAS_OVERLAY_ROUTINES "${_routines}" CACHE INTERNAL
    "BLAS routines provided by the epopenblas overlay for this target")

add_library(${LIB_PREFIX}blas_epopenblas STATIC ${_kernels} ${_helpers})

target_include_directories(${LIB_PREFIX}blas_epopenblas PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_compile_features(${LIB_PREFIX}blas_epopenblas PRIVATE c_std_99)
target_compile_options(${LIB_PREFIX}blas_epopenblas PRIVATE
    -O3
    -ffp-contract=fast
    # Fortran-style complex arithmetic — skip the NaN/Inf branch checks
    # that GCC otherwise emits in __mulxc3/__divxc3.
    -fcx-fortran-rules
)
if(NOT "${EPOPENBLAS_MARCH}" STREQUAL "")
    target_compile_options(${LIB_PREFIX}blas_epopenblas PRIVATE
        -march=${EPOPENBLAS_MARCH})
endif()

if(OpenMP_C_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_epopenblas PUBLIC OpenMP::OpenMP_C)
endif()
# enrm2 / eynrm2 / erotg / yrotg pull in sqrtl + ldexpl from libm.
target_link_libraries(${LIB_PREFIX}blas_epopenblas PUBLIC m)
