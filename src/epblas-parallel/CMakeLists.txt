# Hand-written parallel BLAS overlay — multi-target driver.
#
# For each target in EPBLAS_PARALLEL_BUILT_TARGETS (set by the top-level
# CMakeLists after find_package'ing the eplinalg per-target packages),
# this file sets LIB_PREFIX and add_subdirectory's the matching
# per-target subdir, which produces a STATIC archive
# `${LIB_PREFIX}blas_parallel`.
#
# Per-target subdirs read ${LIB_PREFIX} from outer scope and publish
# their overlay-routines list via the EPBLAS_PARALLEL_OVERLAY_ROUTINES
# variable (set with PARENT_SCOPE inside each per-target subdir).
#
# The WHOLE_ARCHIVE-wrapped composite INTERFACE target
# (`epblas-parallel::<prefix>blas`) is defined in the top-level
# CMakeLists. The renamed `_migrated` IMPORTED archive used by the
# test suite is built in test/CMakeLists.txt (test-only consumer).

# Detect both C and CXX so per-target subdirs (kind10/kind16 in C,
# multifloats in C++) can pick whichever they need.
find_package(OpenMP COMPONENTS C CXX)

# Shared knob across per-target subdirs. Passed verbatim to
# -march=<value> for the overlay kernels.
set(EPBLAS_PARALLEL_MARCH "native" CACHE STRING
    "value passed to -march for parallel BLAS kernels (empty to disable)")

foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(LIB_PREFIX "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_per_target_dir ${CMAKE_CURRENT_SOURCE_DIR}/${_target})
    if(NOT IS_DIRECTORY ${_per_target_dir})
        message(STATUS "epblas-parallel: no kernels for target '${_target}' "
                       "at ${_per_target_dir} — skipping")
        continue()
    endif()
    message(STATUS "epblas-parallel: building ${LIB_PREFIX}blas_parallel "
                   "(target ${_target}) from ${_per_target_dir}")
    add_subdirectory(${_per_target_dir} epblas_parallel_${_target})

    # Capture the per-target overlay-routine list. Each per-target
    # subdir publishes EPBLAS_PARALLEL_OVERLAY_ROUTINES via PARENT_SCOPE
    # — that single global gets overwritten on each iteration, so save
    # a per-target copy here for the migrated-archive helper in
    # test/CMakeLists.txt to consume.
    set(EPBLAS_PARALLEL_OVERLAY_ROUTINES_${_target}
        "${EPBLAS_PARALLEL_OVERLAY_ROUTINES}" PARENT_SCOPE)
endforeach()
