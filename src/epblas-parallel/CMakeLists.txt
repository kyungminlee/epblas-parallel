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
# CACHE INTERNAL var.
#
# The composite INTERFACE target that links overlay-before-serial
# is defined in the top-level CMakeLists.

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
    # subdir publishes EPBLAS_PARALLEL_OVERLAY_ROUTINES (CACHE INTERNAL)
    # — that single global gets overwritten on each iteration, so we
    # save a per-target copy here for tests/epblas-parallel to consume.
    set(EPBLAS_PARALLEL_OVERLAY_ROUTINES_${_target}
        "${EPBLAS_PARALLEL_OVERLAY_ROUTINES}" CACHE INTERNAL
        "Parallel-BLAS overlay routine names for target ${_target}")

    # ── Renamed migrated archive for overlay consistency tests ────
    # `lib${LIB_PREFIX}blas_migrated.a` is the imported serial eplinalg
    # archive (eplinalg::${LIB_PREFIX}blas) with overlay-covered routines
    # renamed `<prefix><routine>_` → `<prefix><routine>_migrated_` via
    # objcopy --redefine-syms. Lets a single test binary link both the
    # composite (overlay's egemm_) AND the renamed archive
    # (egemm_migrated_) and diff them at REAL(KIND=10/16/MF).
    #
    # Used only by tests/epblas-parallel/. Not installed.
    if(EPBLAS_PARALLEL_OVERLAY_ROUTINES)
        set(_redef_file
            ${CMAKE_CURRENT_BINARY_DIR}/${LIB_PREFIX}blas_migrated.redef)
        set(_redef_content "")
        foreach(_r ${EPBLAS_PARALLEL_OVERLAY_ROUTINES})
            string(APPEND _redef_content "${_r}_ ${_r}_migrated_\n")
        endforeach()
        file(GENERATE OUTPUT ${_redef_file} CONTENT "${_redef_content}")

        set(_migrated_archive
            ${CMAKE_CURRENT_BINARY_DIR}/lib${LIB_PREFIX}blas_migrated.a)
        add_custom_command(
            OUTPUT ${_migrated_archive}
            COMMAND ${CMAKE_OBJCOPY} --redefine-syms=${_redef_file}
                    $<TARGET_FILE:eplinalg::${LIB_PREFIX}blas>
                    ${_migrated_archive}
            DEPENDS ${_redef_file}
            COMMENT "objcopy: emitting lib${LIB_PREFIX}blas_migrated.a"
            VERBATIM)
        add_custom_target(${LIB_PREFIX}blas_migrated_build
            DEPENDS ${_migrated_archive})
        add_library(${LIB_PREFIX}blas_migrated STATIC IMPORTED GLOBAL)
        set_target_properties(${LIB_PREFIX}blas_migrated PROPERTIES
            IMPORTED_LOCATION ${_migrated_archive})
        add_dependencies(${LIB_PREFIX}blas_migrated
            ${LIB_PREFIX}blas_migrated_build)
    endif()
endforeach()
