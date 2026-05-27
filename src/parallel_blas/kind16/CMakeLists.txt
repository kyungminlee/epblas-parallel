# Hand-written parallel BLAS — kind16 (REAL(KIND=16) / __float128).
#
# Produces static archive `${LIB_PREFIX}blas_parallel` (= `qblas_parallel`
# for kind16). __float128 has no hardware support on x86-64; every
# add / multiply lowers to a libquadmath call (~hundreds of cycles).
# Arithmetic dominates, so the overlay uses the unblocked Netlib
# reference algorithm with one OpenMP parallel region per call —
# blocking / packing added cost without speedup at this precision.

set(_kernels
    qgemm.c
    xgemm.c
    qgemmtr.c
    xgemmtr.c
    qtrsm.c
    xtrsm.c
    qtrmm.c
    xtrmm.c
    qsyrk.c
    xsyrk.c
    xherk.c
    qsymm.c
    xsymm.c
    xhemm.c
    qsyr2k.c
    xsyr2k.c
    xher2k.c
    qgemv.c
    xgemv.c
    qger.c
    xgeru.c
    xgerc.c
    qsymv.c
    xhemv.c
    qtrsv.c
    xtrsv.c
    qtrmv.c
    xtrmv.c
    qsyr.c
    xher.c
    qsyr2.c
    xher2.c
    # L1 BLAS — subroutines + functions
    qscal.c
    qaxpy.c
    qcopy.c
    qswap.c
    qrot.c
    qdot.c
    qnrm2.c
    qasum.c
    xscal.c
    xqscal.c
    xaxpy.c
    xcopy.c
    xswap.c
    xqrot.c
    xdotu.c
    xdotc.c
    qxnrm2.c
    qxasum.c
    # L2 banded — general
    qgbmv.c
    xgbmv.c
    # L2 banded — symmetric / Hermitian
    qsbmv.c
    xhbmv.c
    # L2 packed — symmetric / Hermitian mat-vec
    qspmv.c
    xhpmv.c
    # L2 banded — triangular mat-vec / solve
    qtbmv.c
    xtbmv.c
    qtbsv.c
    xtbsv.c
    # L2 packed — triangular mat-vec / solve
    qtpmv.c
    xtpmv.c
    qtpsv.c
    xtpsv.c
    # L2 packed — sym/Herm rank-1 + rank-2
    qspr.c
    xhpr.c
    qspr2.c
    xhpr2.c
    # L1 misc — cabs1, iamax, Givens generators
    qcabs1.c
    iqamax.c
    ixamax.c
    qrotg.c
    qrotm.c
    qrotmg.c
    xrotg.c
)

set(_routines "")
foreach(_k ${_kernels})
    string(REGEX REPLACE "\\.c$" "" _r "${_k}")
    list(APPEND _routines "${_r}")
endforeach()
set(PARALLEL_BLAS_OVERLAY_ROUTINES "${_routines}" CACHE INTERNAL
    "BLAS routines provided by the parallel overlay for this target")

add_library(${LIB_PREFIX}blas_parallel STATIC ${_kernels})
add_library(epparablas::${LIB_PREFIX}blas_parallel ALIAS ${LIB_PREFIX}blas_parallel)

target_compile_features(${LIB_PREFIX}blas_parallel PRIVATE c_std_99)
target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
    -O3
    -ffp-contract=fast
    -fcx-fortran-rules  # see note in kind10/CMakeLists.txt
)
if(NOT "${PARALLEL_BLAS_MARCH}" STREQUAL "")
    target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
        -march=${PARALLEL_BLAS_MARCH})
endif()

# Quadmath provides __float128 arithmetic + the complex helpers
# (conjq, cabsq, ...). Link explicitly — libquadmath is a separate .so
# from libgcc.
target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC quadmath)

if(OpenMP_C_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC OpenMP::OpenMP_C)
endif()
