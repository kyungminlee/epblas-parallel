# Overlay test suites — per-target consistency / fuzz / bench /
# perf drivers for the hand-written kernels.

find_program(FYPP fypp REQUIRED)

# Helper for per-target wrapper shims:
#   add_fypp_source(<out_var> <shim.fypp> <body_dir>)
# generates <build>/<target_subdir>/<basename>.f90 from <shim.fypp>,
# with -I pointing at <body_dir> so #:include resolves the shared body
# file. The shim's parent directory name (e.g. `target_kind10`) is
# carried into the build output so per-target shims of the same name
# (every target has its own `fuzz_util.fypp`) don't collide on a
# shared OUTPUT path.
function(add_fypp_source out_var shim_path body_dir)
    get_filename_component(_stem ${shim_path} NAME_WE)
    get_filename_component(_shim_parent ${shim_path} DIRECTORY)
    get_filename_component(_target_subdir ${_shim_parent} NAME)
    set(_out ${CMAKE_CURRENT_BINARY_DIR}/${_target_subdir}/${_stem}.f90)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_target_subdir})
    # Depend on every .fypp in body_dir so edits to the shared body
    # (or any future #:include sibling) trigger regeneration.
    file(GLOB _body_fypps CONFIGURE_DEPENDS ${body_dir}/*.fypp)
    add_custom_command(
        OUTPUT ${_out}
        COMMAND ${FYPP} -I ${body_dir} ${shim_path} ${_out}
        DEPENDS ${shim_path} ${_body_fypps}
        COMMENT "fypp ${shim_path} -> ${_out}"
        VERBATIM
    )
    set(${out_var} ${_out} PARENT_SCOPE)
endfunction()

# ── Renamed-migrated archives (test/bench-only) ──────────────────
# For each TESTABLE target we objcopy-rename overlay-covered Fortran
# symbols `<r>_` → `<r>_migrated_` in the imported eplinalg per-target
# archive. A single test binary then links the composite (overlay's
# `<r>_`) and the renamed archive (`<r>_migrated_`) and diffs them.
# Two IMPORTED archives per target:
#   ${prefix}blas_migrated         — renames EPBLAS_PARALLEL_OVERLAY_ROUTINES_${target}
#                                    (consumed by tests/epblas-parallel/)
#   ${prefix}blas_openblas_migrated — renames EPBLAS_OPENBLAS_OVERLAY_ROUTINES_${target}
#                                    (consumed by tests/epblas-openblas/, when the
#                                     epblas-openblas overlay exists for that target)
function(_epblas_make_migrated_archive arch_name routines)
    if(NOT routines)
        return()
    endif()
    set(_redef_file ${CMAKE_CURRENT_BINARY_DIR}/${arch_name}.redef)
    set(_redef_content "")
    foreach(_r ${routines})
        string(APPEND _redef_content "${_r}_ ${_r}_migrated_\n")
    endforeach()
    file(GENERATE OUTPUT ${_redef_file} CONTENT "${_redef_content}")

    set(_archive ${CMAKE_CURRENT_BINARY_DIR}/lib${arch_name}.a)
    add_custom_command(
        OUTPUT ${_archive}
        COMMAND ${CMAKE_OBJCOPY} --redefine-syms=${_redef_file}
                $<TARGET_FILE:eplinalg::${LIB_PREFIX}blas>
                ${_archive}
        DEPENDS ${_redef_file}
        COMMENT "objcopy: emitting lib${arch_name}.a"
        VERBATIM)
    add_custom_target(${arch_name}_build DEPENDS ${_archive})
    add_library(${arch_name} STATIC IMPORTED GLOBAL)
    set_target_properties(${arch_name} PROPERTIES IMPORTED_LOCATION ${_archive})
    add_dependencies(${arch_name} ${arch_name}_build)
endfunction()

foreach(_target IN LISTS EPBLAS_PARALLEL_TESTABLE_TARGETS)
    set(LIB_PREFIX "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    _epblas_make_migrated_archive(${LIB_PREFIX}blas_migrated
        "${EPBLAS_PARALLEL_OVERLAY_ROUTINES_${_target}}")
    if(TARGET ${LIB_PREFIX}blas_openblas)
        _epblas_make_migrated_archive(${LIB_PREFIX}blas_openblas_migrated
            "${EPBLAS_OPENBLAS_OVERLAY_ROUTINES_${_target}}")
    endif()
endforeach()

add_subdirectory(epblas-parallel)
add_subdirectory(epblas-openblas)

# ── End-to-end perf sweeps (label `sweep`, ~30 min–hours) ─────────
# Registered as ctest tests so the `sweep` and `e2e` workflow presets
# can pull them in. Default `ctest` invocations should exclude
# `-L sweep` (or use `--preset fuzz`) — they're long-running.
add_test(NAME cmp5_sweep
    COMMAND bash ${PROJECT_SOURCE_DIR}/bench/cmp5/run_cmp5.sh)
set_tests_properties(cmp5_sweep PROPERTIES
    ENVIRONMENT "STAGE_E=${PROJECT_BINARY_DIR}"
    LABELS "sweep"
    TIMEOUT 7200)

add_test(NAME perf_sweep
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/run_perf_sweep.sh
            ${PROJECT_BINARY_DIR})
set_tests_properties(perf_sweep PROPERTIES
    LABELS "sweep"
    TIMEOUT 7200)
