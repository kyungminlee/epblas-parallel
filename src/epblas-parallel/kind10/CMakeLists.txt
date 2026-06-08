# Hand-written parallel BLAS — kind10 (REAL(KIND=10) / 80-bit long double).
#
# Produces a STATIC archive `${LIB_PREFIX}blas_parallel` (= `eblas_parallel`
# for kind10) that supplies overlay implementations of selected BLAS
# entry points. Step 0 ships only a passthrough stub for egemm_ that
# forwards to the migrated archive's renamed symbol `egemm_migrated_`
# (produced by the objcopy rename in step 0b). This proves the
# Fortran-ABI C boundary end-to-end before any real kernel work.
#
# Linking model: see doc/design.md §1. The top-level CMakeLists.txt
# wraps this archive in an INTERFACE composite that links `_parallel`
# WHOLE_ARCHIVE before the serial migrated archive so overlay symbols
# win. This file builds the `_parallel` half only.

set(_kernels
    egemv.c
    ygemv.c
    eger.c
    ygeru.c
    ygerc.c
    esymv.c
    yhemv.c
    etrsv.c
    ytrsv.c
    etrmv.c
    ytrmv.c
    esyr.c
    yher.c
    esyr2.c
    yher2.c
    # L1 BLAS — subroutines + functions
    escal.c
    eaxpy.c
    ecopy.c
    eswap.c
    erot.c
    edot.c
    enrm2.c
    easum.c
    yscal.c
    yescal.c
    yaxpy.c
    ycopy.c
    yswap.c
    yerot.c
    ydotu.c
    ydotc.c
    eynrm2.c
    eyasum.c
    # L2 banded — general
    egbmv.c
    ygbmv.c
    # L2 banded — symmetric / Hermitian
    esbmv.c
    yhbmv.c
    # L2 packed — symmetric / Hermitian mat-vec
    espmv.c
    yhpmv.c
    # L2 banded — triangular mat-vec / solve
    etbmv.c
    ytbmv.c
    etbsv.c
    ytbsv.c
    # L2 packed — triangular mat-vec / solve
    etpmv.c
    ytpmv.c
    etpsv.c
    ytpsv.c
    # L2 packed — sym/Herm rank-1 + rank-2
    espr.c
    yhpr.c
    espr2.c
    yhpr2.c
    # L1 misc — cabs1, iamax, Givens generators
    ecabs1.c
    ieamax.c
    iyamax.c
    erotg.c
    erotm.c
    erotmg.c
    yrotg.c
)

# L3 routines are each split across two translation units: the public
# Fortran entry (parallel driver, *_parallel.c) and an internal
# single-thread worker (*_serial.c). A *_serial worker is (a) reused by the
# L3 routines that call it from inside their own parallel region (egemm/
# ygemm trailing updates), and (b) the fully-serial path its own driver
# delegates to under the omp_in_parallel() nesting guard — opening a nested
# team would trip the libgomp barrier wedge.
# Only the bare routine names (egemm, ygemm, etrsm, …) are Fortran-callable
# overlays — the _serial workers are internal — so these sources are added
# to the build but NOT name-derived into the routine list; the routine names
# are appended by hand below (mirrors the openblas overlay's `_helpers`).
set(_split_sources
    egemm_parallel.c
    egemm_serial.c
    etri_kernel.c
    etrsm_parallel.c
    etrsm_serial.c
    etrsm_kernel.c
    etrsm_pack.c
    etrmm_kernel.c
    etrmm_pack.c
    ygemm_parallel.c
    ygemm_serial.c
    ytrsm_parallel.c
    ytrsm_serial.c
    ytrmm_parallel.c
    ytrmm_serial.c
    ysyrk_parallel.c
    ysyrk_serial.c
    yherk_parallel.c
    yherk_serial.c
    ysyr2k_parallel.c
    ysyr2k_serial.c
    yher2k_parallel.c
    yher2k_serial.c
    ysymm_parallel.c
    ysymm_serial.c
    yhemm_parallel.c
    yhemm_serial.c
    ygemmtr_parallel.c
    ygemmtr_serial.c
    egemmtr_parallel.c
    egemmtr_serial.c
    etrmm_parallel.c
    etrmm_serial.c
    esyrk_parallel.c
    esyrk_serial.c
    esyr2k_parallel.c
    esyr2k_serial.c
    esymm_parallel.c
    esymm_serial.c
)

# Publish the overlay-covered routine names so the BLAS section in
# cmake/CMakeLists.txt knows what to rename in the consistency archive.
# Strip the .c suffix from each kernel filename; the split routines
# (egemm, etrsm, ygemm) are appended by hand (their file basenames don't
# match the routine name — see above).
set(_routines "")
foreach(_k ${_kernels})
    string(REGEX REPLACE "\\.c$" "" _r "${_k}")
    list(APPEND _routines "${_r}")
endforeach()
list(APPEND _routines egemm etrsm ygemm ytrsm ytrmm ysyrk yherk ysyr2k yher2k ysymm yhemm ygemmtr egemmtr etrmm esyrk esyr2k esymm)
set(EPBLAS_PARALLEL_OVERLAY_ROUTINES "${_routines}" PARENT_SCOPE)

add_library(${LIB_PREFIX}blas_parallel STATIC ${_kernels} ${_split_sources})
add_library(epblas-parallel::${LIB_PREFIX}blas_parallel ALIAS ${LIB_PREFIX}blas_parallel)

target_compile_features(${LIB_PREFIX}blas_parallel PRIVATE c_std_99)
target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
    -O3
    -ffp-contract=fast
    # Force 16-byte hot-loop alignment. The int->ptrdiff_t index conversion
    # widens integer instruction encodings by one byte each (incl->incq,
    # cmpl->cmpq, movslq, REX prefixes), which shifts GCC's default nop-padding
    # and lands a few hot loops off a 16-byte boundary (ieamax mod16 0->8,
    # eynrm2 loop-body fetch). -falign-loops=16 pins them back, restoring
    # parity; it is neutral on the routines that were already aligned and worse
    # at 32 (over-align).
    -falign-loops=16
    # Use Fortran-style complex arithmetic (skip NaN/Inf checks).
    # Without this, gcc emits __mulxc3/__divxc3 library calls for
    # _Complex long double — measurably slower than the gfortran
    # path used by the migrated archive (e.g. ytrsm OMP=1 was 30%
    # behind migrated until this flag landed).
    -fcx-fortran-rules
)
# -march comes from src/epblas-parallel/CMakeLists.txt (shared knob).
if(NOT "${EPBLAS_PARALLEL_MARCH}" STREQUAL "")
    target_compile_options(${LIB_PREFIX}blas_parallel PRIVATE
        -march=${EPBLAS_PARALLEL_MARCH})
endif()

# OpenMP comes in at step 4. Wire it now so the per-kernel #pragmas
# compile from the start.
if(OpenMP_C_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_parallel PUBLIC OpenMP::OpenMP_C)
endif()

# Symbols this archive REFERS to (and the future composite resolves):
# the migrated archive renamed via objcopy in step 0b. We don't link
# anything here — the static archive can carry unresolved externals;
# they're satisfied at executable link time by the composite.
