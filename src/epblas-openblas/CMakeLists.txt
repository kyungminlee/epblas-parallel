# OpenBLAS-derived BLAS overlay (epblas-openblas) — multi-target driver.
#
# Builds a separate STATIC archive `${LIB_PREFIX}blas_openblas` per
# target whose per-target kernel set exists under
# src/epblas-openblas/<target>/. This archive is NOT wired into the
# public `epblas-parallel::<prefix>blas` composite — it exists purely
# for A/B comparisons against the migrated baseline (via
# test/epblas-openblas/).
#
# All three targets (kind10, kind16, multifloats) have kernel sets;
# a target is picked up automatically once its <target>/ subdir exists.

find_package(OpenMP COMPONENTS C)

set(EPBLAS_OPENBLAS_MARCH "native" CACHE STRING
    "value passed to -march for epblas-openblas kernels (empty to disable)")

foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(LIB_PREFIX "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_per_target_dir ${CMAKE_CURRENT_SOURCE_DIR}/${_target})
    if(NOT IS_DIRECTORY ${_per_target_dir})
        message(STATUS "epblas-openblas: no kernels for target '${_target}' "
                       "at ${_per_target_dir} — skipping")
        continue()
    endif()
    message(STATUS "epblas-openblas: building ${LIB_PREFIX}blas_openblas "
                   "(target ${_target}) from ${_per_target_dir}")
    add_subdirectory(${_per_target_dir} epblas_openblas_${_target})

    # Capture the per-target openblas overlay-routine list. The
    # per-target subdir publishes EPBLAS_OPENBLAS_OVERLAY_ROUTINES
    # via PARENT_SCOPE; that single global is overwritten between
    # target iterations, so save a per-target copy here for the
    # migrated-archive helper in test/CMakeLists.txt to consume.
    set(EPBLAS_OPENBLAS_OVERLAY_ROUTINES_${_target}
        "${EPBLAS_OPENBLAS_OVERLAY_ROUTINES}" PARENT_SCOPE)
endforeach()
