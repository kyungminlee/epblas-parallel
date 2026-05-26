# Hand-written parallel BLAS overlay — multi-target driver.
#
# For each target in EPBLAS_BUILT_TARGETS (set by the top-level
# CMakeLists after find_package'ing the migrator's per-target
# packages), this file sets LIB_PREFIX and add_subdirectory's the
# matching per-target subdir, which produces a STATIC archive
# `${LIB_PREFIX}blas_parallel`.
#
# Per-target subdirs are unchanged from when they lived inside
# fortran-migrator/src/parallel_blas/<target>/ — they still read
# ${LIB_PREFIX} from outer scope.
#
# The composite INTERFACE target that links overlay-before-serial
# is defined separately in step 4.

# Detect both C and CXX so per-target subdirs (kind10/kind16 in C,
# multifloats in C++) can pick whichever they need.
find_package(OpenMP COMPONENTS C CXX)

# Shared knob across per-target subdirs. Passed verbatim to
# -march=<value> for the overlay kernels.
set(PARALLEL_BLAS_MARCH "native" CACHE STRING
    "value passed to -march for parallel BLAS kernels (empty to disable)")

foreach(_target IN LISTS EPBLAS_BUILT_TARGETS)
    set(LIB_PREFIX "${EPBLAS_PREFIX_${_target}}")
    set(_per_target_dir ${CMAKE_CURRENT_SOURCE_DIR}/${_target})
    if(NOT IS_DIRECTORY ${_per_target_dir})
        message(STATUS "parallel_blas: no kernels for target '${_target}' "
                       "at ${_per_target_dir} — skipping")
        continue()
    endif()
    message(STATUS "parallel_blas: building ${LIB_PREFIX}blas_parallel "
                   "(target ${_target}) from ${_per_target_dir}")
    add_subdirectory(${_per_target_dir} parallel_blas_${_target})
endforeach()
