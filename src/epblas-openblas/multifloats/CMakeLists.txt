# epblas-openblas — multifloats (double-double, multifloats::float64x2 /
# std::complex<float64x2>).
#
# OpenBLAS D and Z source ports, retyped to the multifloats double-double
# type. Faithful hand-port of the kind10 tree (same algorithm, threading,
# and thresholds — only the element type and its libm/literal surface
# differ); the port conventions are documented in
# src/epblas-openblas/CHECKLIST.md.
#
# C++ (not C) because the double-double type is a C++ struct with operator
# overloads; complex arithmetic runs on the layout-compatible
# std::complex<float64x2> (reinterpret_cast at the complex64x2 ABI
# boundary). Built with -ffp-contract=on (NOT =fast — fast folds across the
# error-free two_prod/two_sum transforms) and WITHOUT -ffast-math.
#
# Builds a STATIC archive `${LIB_PREFIX}blas_openblas` (mwblas_openblas)
# consumed only by test/epblas-openblas/. NOT wired into the public
# `epblas-parallel::${LIB_PREFIX}blas` composite.
#
# Adding a routine:
#   1. Drop the ported source into this directory, e.g. waxpy.cpp
#   2. Append the filename to _kernels below.
#   3. The matching test/epblas-openblas/.../target_multifloats/ shim picks
#      it up automatically.

# The master driver finds only OpenMP::OpenMP_C; the DD kernels are C++.
find_package(OpenMP COMPONENTS CXX)

set(_kernels
    # ── L1 real ──
    maxpy.cpp
    mcopy.cpp
    mswap.cpp
    mscal.cpp
    mdot.cpp
    mnrm2.cpp
    masum.cpp
    imamax.cpp
    mrot.cpp
    mrotg.cpp
    mrotm.cpp
    mrotmg.cpp
    # ── L1 complex ──
    waxpy.cpp
    wcopy.cpp
    wswap.cpp
    wscal.cpp
    wmscal.cpp
    wdotu.cpp
    wdotc.cpp
    mwnrm2.cpp
    mwasum.cpp
    wmrot.cpp
    wrotg.cpp
    iwamax.cpp
    mcabs1.cpp
    # ── L2 real ──
    mgemv.cpp
    mger.cpp
    msymv.cpp
    msyr.cpp
    msyr2.cpp
    mspr.cpp
    mspr2.cpp
    mspmv.cpp
    msbmv.cpp
    mgbmv.cpp
    mtrmv.cpp
    mtrsv.cpp
    mtbmv.cpp
    mtbsv.cpp
    mtpmv.cpp
    mtpsv.cpp
    # ── L2 complex (general / triangular / band / packed) ──
    wgemv.cpp
    wgeru.cpp
    wgerc.cpp
    wgbmv.cpp
    wtrmv.cpp
    wtrsv.cpp
    wtbmv.cpp
    wtbsv.cpp
    wtpmv.cpp
    wtpsv.cpp
    # ── L2 complex Hermitian ──
    whemv.cpp
    whbmv.cpp
    wher.cpp
    wher2.cpp
    whpmv.cpp
    whpr.cpp
    whpr2.cpp
    # ── L3 real ──
    mgemm.cpp
    msymm.cpp
    msyrk.cpp
    msyr2k.cpp
    mtrmm.cpp
    mtrsm.cpp
    mgemmtr.cpp
    # ── L3 complex ──
    wgemm.cpp
    wsymm.cpp
    wsyrk.cpp
    wsyr2k.cpp
    wtrmm.cpp
    wtrsm.cpp
    wgemmtr.cpp
    whemm.cpp
    wherk.cpp
    wher2k.cpp
)

# Internal helper sources — shared L3 kernel + packers (DD ports of the
# kind10 eblas_l3_{real,complex}.c). They contribute symbols to the archive
# but are not Fortran-callable, so they're listed separately from `_kernels`
# and NOT included in `EPBLAS_OPENBLAS_OVERLAY_ROUTINES`.
set(_helpers
    ../common/mblas_l3_real.cpp
    ../common/mblas_l3_complex.cpp
)

# Publish the overlay-covered routine set so the driver in
# src/epblas-openblas/CMakeLists.txt knows which symbols to objcopy-rename
# in the matching `${LIB_PREFIX}blas_openblas_migrated` archive.
set(_routines "")
foreach(_k ${_kernels})
    string(REGEX REPLACE "\\.cpp$" "" _r "${_k}")
    list(APPEND _routines "${_r}")
endforeach()
set(EPBLAS_OPENBLAS_OVERLAY_ROUTINES "${_routines}" PARENT_SCOPE)

add_library(${LIB_PREFIX}blas_openblas STATIC ${_kernels} ${_helpers})
add_library(epblas-openblas::${LIB_PREFIX}blas_openblas ALIAS ${LIB_PREFIX}blas_openblas)

set_target_properties(${LIB_PREFIX}blas_openblas PROPERTIES
    EXPORT_NAME ${LIB_PREFIX}blas)

target_include_directories(${LIB_PREFIX}blas_openblas PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_compile_features(${LIB_PREFIX}blas_openblas PRIVATE cxx_std_17)
target_compile_options(${LIB_PREFIX}blas_openblas PRIVATE
    -O3
    # NOT =fast: fast contracts across statements and breaks the
    # error-free two_prod/two_sum transforms the DD type relies on.
    -ffp-contract=on
)
if(NOT "${EPBLAS_OPENBLAS_MARCH}" STREQUAL "")
    target_compile_options(${LIB_PREFIX}blas_openblas PRIVATE
        -march=${EPBLAS_OPENBLAS_MARCH})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIB_PREFIX}blas_openblas PUBLIC OpenMP::OpenMP_CXX)
endif()
# multifloats supplies the float64x2 type + its libm surface (sqrtdd,
# ldexpdd, c*dd, …); m for any residual double-precision libm calls.
target_link_libraries(${LIB_PREFIX}blas_openblas PUBLIC multifloats m)
