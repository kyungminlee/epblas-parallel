# OpenBLAS-derived BLAS overlay (epblas-openblas) — multi-target driver.
#
# Builds a separate STATIC archive `${LIB_PREFIX}blas_openblas` per
# target whose per-target kernel set exists under
# src/epblas-openblas/<target>/. This archive is NOT wired into the
# public `epblas-parallel::<prefix>blas` composite — it exists purely
# for A/B comparisons against the migrated baseline (via
# tests/epblas-openblas/).
#
# Initial target: kind10. kind16 / multifloats can be added later by
# creating sibling target_<name>/ subdirs.

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
    # (CACHE INTERNAL); that single global is overwritten between
    # target iterations, so we save a per-target copy here for the
    # test suite + the renamed-migrated archive below.
    set(EPBLAS_OPENBLAS_OVERLAY_ROUTINES_${_target}
        "${EPBLAS_OPENBLAS_OVERLAY_ROUTINES}" CACHE INTERNAL
        "epblas-openblas overlay routine names for target ${_target}")

    # ── Renamed migrated archive for openblas A/B tests ───────────
    # `lib${LIB_PREFIX}blas_openblas_migrated.a` is the imported
    # serial eplinalg archive (eplinalg::${LIB_PREFIX}blas) with
    # openblas-covered routines renamed
    # `<prefix><routine>_` → `<prefix><routine>_migrated_` via
    # objcopy --redefine-syms. Separate from the epblas-parallel
    # _migrated archive because the openblas routine set is
    # independent (may diverge as the port progresses).
    if(EPBLAS_OPENBLAS_OVERLAY_ROUTINES)
        set(_redef_file
            ${CMAKE_CURRENT_BINARY_DIR}/${LIB_PREFIX}blas_openblas_migrated.redef)
        set(_redef_content "")
        foreach(_r ${EPBLAS_OPENBLAS_OVERLAY_ROUTINES})
            string(APPEND _redef_content "${_r}_ ${_r}_migrated_\n")
        endforeach()
        file(GENERATE OUTPUT ${_redef_file} CONTENT "${_redef_content}")

        set(_migrated_archive
            ${CMAKE_CURRENT_BINARY_DIR}/lib${LIB_PREFIX}blas_openblas_migrated.a)
        add_custom_command(
            OUTPUT ${_migrated_archive}
            COMMAND ${CMAKE_OBJCOPY} --redefine-syms=${_redef_file}
                    $<TARGET_FILE:eplinalg::${LIB_PREFIX}blas>
                    ${_migrated_archive}
            DEPENDS ${_redef_file}
            COMMENT "objcopy: emitting lib${LIB_PREFIX}blas_openblas_migrated.a"
            VERBATIM)
        add_custom_target(${LIB_PREFIX}blas_openblas_migrated_build
            DEPENDS ${_migrated_archive})
        add_library(${LIB_PREFIX}blas_openblas_migrated STATIC IMPORTED GLOBAL)
        set_target_properties(${LIB_PREFIX}blas_openblas_migrated PROPERTIES
            IMPORTED_LOCATION ${_migrated_archive})
        add_dependencies(${LIB_PREFIX}blas_openblas_migrated
            ${LIB_PREFIX}blas_openblas_migrated_build)
    endif()
endforeach()
