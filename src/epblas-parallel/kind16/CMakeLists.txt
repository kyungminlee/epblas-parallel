# Hand-written parallel BLAS — kind16 (REAL(KIND=16) / __float128).
#
# Produces static archive `${LIB_PREFIX}blas_parallel` (= `qxblas_parallel`
# for kind16). __float128 has no hardware support on x86-64; every
# add / multiply lowers to a libquadmath call (~hundreds of cycles).
# Arithmetic dominates, so the overlay uses the unblocked Netlib
# reference algorithm with one OpenMP parallel region per call —
# blocking / packing added cost without speedup at this precision.

set(_kernels
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

# L3 routines are each split across two translation units (mirroring
# kind10): the public Fortran entry (parallel driver, *_parallel.c) and an
# internal single-thread worker (*_serial.c). The *_serial worker is (a)
# reused by L3 routines that call it from inside their own parallel region
# (qtrsm/xtrsm trailing GEMM updates call qgemm_serial_/xgemm_serial_), and
# (b) the fully-serial path its own driver delegates to under the
# omp_in_parallel() nesting guard. The real-quad routines pack inline in
# their *_serial.c; the complex-quad routines xgemm/xsymm/xhemm share the
# packed substrate xl3_complex.c (a faithful par-side copy of the
# epblas-openblas common substrate, mirroring kind10's etri_kernel.c).
# xherk/xsyrk do not use the substrate directly — they route their trailing
# rank-k update through xgemm_serial_.
# Only the bare routine names (qgemm, xgemm, …) are Fortran-callable
# overlays, so these sources are added to the build but NOT name-derived;
# the names are appended by hand below (mirrors kind10).
set(_split_sources
    qtri_kernel.c
    xl3_complex.c
    qgemm_parallel.c
    qgemm_serial.c
    xgemm_parallel.c
    xgemm_serial.c
    qgemmtr_parallel.c
    qgemmtr_serial.c
    xgemmtr_parallel.c
    xgemmtr_serial.c
    qtrsm_parallel.c
    qtrsm_serial.c
    xtrsm_parallel.c
    xtrsm_serial.c
    xtri_driver.c
    qtrmm_parallel.c
    qtrmm_serial.c
    xtrmm_parallel.c
    xtrmm_serial.c
    qsyrk_parallel.c
    qsyrk_serial.c
    xsyrk_parallel.c
    xsyrk_serial.c
    xherk_parallel.c
    xherk_serial.c
    qsymm_parallel.c
    qsymm_serial.c
    xsymm_parallel.c
    xsymm_serial.c
    xhemm_parallel.c
    xhemm_serial.c
    qsyr2k_parallel.c
    qsyr2k_serial.c
    xsyr2k_parallel.c
    xsyr2k_serial.c
    xher2k_parallel.c
    xher2k_serial.c
)

epblas_kernel_routines(_routines ${_kernels})
list(APPEND _routines qgemm xgemm qgemmtr xgemmtr qtrsm xtrsm qtrmm xtrmm
                      qsyrk xsyrk xherk qsymm xsymm xhemm qsyr2k xsyr2k xher2k)
set(EPBLAS_PARALLEL_OVERLAY_ROUTINES "${_routines}" PARENT_SCOPE)

# Archive + alias + c_std_99 + [-O3 <flags>] + guarded -march, via the
# shared helper (cmake/EpblasKindHelpers.cmake).
epblas_add_kind_archive(${LIB_PREFIX}blas_parallel
    NAMESPACE epblas-parallel
    LANGUAGE C
    MARCH "${EPBLAS_PARALLEL_MARCH}"
    SOURCES ${_kernels} ${_split_sources}
    COMPILE_OPTIONS
        -ffp-contract=fast
        -fcx-fortran-rules  # see note in kind10/CMakeLists.txt
)

# qtrmm's RLT inner AXPY loop landed at a 32B-chunk offset of 0x18 and
# straddled a fetch boundary; the instruction-identical gfortran reference
# loop sits at 0x0 and ran ~3% faster (par-RLT was the only serial cell
# above the bar while par-RUT, same body at offset 0x8, passed). Pinning
# loop heads to 32B removes the placement lottery (see kind10/enrm2).
set_source_files_properties(qtrmm_serial.c PROPERTIES COMPILE_OPTIONS "-falign-loops=32")

# Quadmath provides __float128 arithmetic + the complex helpers
# (conjq, cabsq, ...). Link explicitly — libquadmath is a separate .so
# from libgcc.
target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC quadmath)

if(OpenMP_C_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC OpenMP::OpenMP_C)
endif()
