# epblas-openblas — kind16 (REAL(KIND=16) / 128-bit __float128).
#
# OpenBLAS D and Z source ports, retyped to __float128 / __complex128.
# Faithful hand-port of the kind10 tree (same algorithm, threading, and
# thresholds — only the type and its libm/literal surface differ); the
# substitution set is documented in src/epblas-openblas/CHECKLIST.md.
#
# Builds a STATIC archive `${LIB_PREFIX}blas_openblas` consumed only by
# test/epblas-openblas/. NOT wired into the public
# `epblas-parallel::${LIB_PREFIX}blas` composite.
#
# Adding a routine:
#   1. Drop the ported source into this directory, e.g. qaxpy.c
#   2. Append the filename to _kernels below.
#   3. The matching test/epblas-openblas/.../target_kind16/ shim picks it up
#      automatically — see CHECKLIST.md for the per-routine status table.

set(_kernels
    # ── L1 real ──
    qaxpy.c
    qcopy.c
    qswap.c
    qscal.c
    qdot.c
    qnrm2.c
    qasum.c
    iqamax.c
    qrot.c
    qrotg.c
    qrotm.c
    qrotmg.c
    # ── L1 complex ──
    xaxpy.c
    xcopy.c
    xswap.c
    xscal.c
    xqscal.c
    xdotu.c
    xdotc.c
    qxnrm2.c
    qxasum.c
    xqrot.c
    xrotg.c
    ixamax.c
    qcabs1.c
    # ── L2 real ──
    qgemv.c
    qger.c
    qsymv.c
    qsyr.c
    qsyr2.c
    qspr.c
    qspr2.c
    qspmv.c
    qsbmv.c
    qgbmv.c
    qtrmv.c
    qtrsv.c
    qtbmv.c
    qtbsv.c
    qtpmv.c
    qtpsv.c
    # ── L2 complex (general / triangular / band / packed) ──
    xgemv.c
    xgeru.c
    xgerc.c
    xgbmv.c
    xtrmv.c
    xtrsv.c
    xtbmv.c
    xtbsv.c
    xtpmv.c
    xtpsv.c
    # ── L2 complex Hermitian ──
    xhemv.c
    xhbmv.c
    xher.c
    xher2.c
    xhpmv.c
    xhpr.c
    xhpr2.c
    # ── L3 real ──
    qgemm.c
    qsymm.c
    qsyrk.c
    qsyr2k.c
    qtrmm.c
    qtrsm.c
    qgemmtr.c
    # ── L3 complex ──
    xgemm.c
    xsymm.c
    xsyrk.c
    xsyr2k.c
    xtrmm.c
    xtrsm.c
    xgemmtr.c
    xhemm.c
    xherk.c
    xher2k.c
)

# Internal helper sources — shared L3 kernel + packers (quad ports of the
# kind10 eblas_l3_{real,complex}.c). They contribute symbols to the archive
# but are not Fortran-callable, so they're listed separately from `_kernels`
# and NOT included in `EPBLAS_OPENBLAS_OVERLAY_ROUTINES`.
set(_helpers
    ../common/qblas_l3_real.c
    ../common/qblas_l3_complex.c
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
# sqrtq / ldexpq / conjq / crealq / cabsq come from libquadmath; m for any
# residual double-precision libm calls (e.g. ceil in the L3 block sizing).
target_link_libraries(${LIB_PREFIX}blas_openblas PUBLIC quadmath m)
