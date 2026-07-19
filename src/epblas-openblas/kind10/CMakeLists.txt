# epblas-openblas — kind10 (REAL(KIND=10) / 80-bit long double).
#
# OpenBLAS D and Z source ports, retyped to long double / _Complex long double.
# Builds a STATIC archive `${LIB_PREFIX}blas_openblas` consumed only by
# test/epblas-openblas/. NOT wired into the public
# `epblas-parallel::${LIB_PREFIX}blas` composite.
#
# Adding a routine:
#   1. Drop the ported source into this directory, e.g. eaxpy.c
#   2. Append the filename to _kernels below.
#   3. The matching test/epblas-openblas/.../target_kind10/ shim picks it up
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
# `EPBLAS_OPENBLAS_OVERLAY_ROUTINES`.
set(_helpers
    ../common/eblas_l3_real.c
    ../common/eblas_l3_complex.c
)

# Publish the overlay-covered routine set so the driver in
# src/epblas-openblas/CMakeLists.txt knows which symbols to objcopy-rename
# in the matching `${LIB_PREFIX}blas_openblas_migrated` archive.
epblas_kernel_routines(_routines ${_kernels})
set(EPBLAS_OPENBLAS_OVERLAY_ROUTINES "${_routines}" PARENT_SCOPE)

# Archive + alias + c_std_99 + [-O3 <flags>] + guarded -march, via the
# shared helper (cmake/EpblasKindHelpers.cmake).
# Consumer-facing EXPORT_NAME is bare `${LIB_PREFIX}blas` — the
# `epblas-openblas::` namespace already disambiguates from the parallel
# package's composite. Internal target name keeps the `_openblas`
# suffix because CMake requires per-project target uniqueness and the
# parallel package's composite target is named `${LIB_PREFIX}blas`.
epblas_add_kind_archive(${LIB_PREFIX}blas_openblas
    NAMESPACE epblas-openblas
    LANGUAGE C
    MARCH "${EPBLAS_OPENBLAS_MARCH}"
    EXPORT_NAME ${LIB_PREFIX}blas
    COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common
    SOURCES ${_kernels} ${_helpers}
    COMPILE_OPTIONS
        -ffp-contract=fast
        # Fortran-style complex arithmetic — skip the NaN/Inf branch checks
        # that GCC otherwise emits in __mulxc3/__divxc3.
        -fcx-fortran-rules
)

if(OpenMP_C_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_openblas PUBLIC OpenMP::OpenMP_C)
endif()
# enrm2 / eynrm2 / erotg / yrotg pull in sqrtl + ldexpl from libm.
target_link_libraries(${LIB_PREFIX}blas_openblas PUBLIC m)
