# Parallel BLAS overlay test suite — multi-target driver.
#
# Per target in EPBLAS_PARALLEL_TESTABLE_TARGETS we generate:
#   common/target_<name>/fuzz_util.fypp        → fuzz_util.f90 (per-target shim)
#   consistency/target_<name>/fuzz_*.fypp      → fuzz drivers (overlay vs migrated)
#
# The C/C++ perf harnesses live in benchmark/drivers/target_<name>/perf_*.c|.cpp
# (shared with the epblas-openblas suite); see _perf_target_dir below.
#
# The legacy Fortran bench harness (bench/target_<name>/bench_*.fypp) was
# retired (archived locally under workspace/archive/tests/) — the C perf
# harness is the supported timing path and reports bare wall time (ns/call).
#
# Each test binary links the composite `epblas-parallel::<prefix>blas`
# (WHOLE_ARCHIVE-wrapped overlay) AND the renamed `_migrated` archive
# (`<prefix>blas_migrated` — produced by test/CMakeLists.txt) so a
# single driver can call both `egemm_` and `egemm_migrated_` and
# diff them.
#
# `eplinalg::blas` supplies lsame_/xerbla_: the migrated archive is
# the per-precision eplinalg archive (with overlay routines renamed)
# and references them as U-symbols; the production composite no
# longer drags eplinalg::blas in transitively, so the link line lists
# it explicitly.

# multifloats brings in the Fortran-side module needed by the
# multifloats target's fuzz_util.
set(_have_multifloats_f FALSE)
if(TARGET multifloatsf)
    set(_have_multifloats_f TRUE)
endif()

foreach(_target IN LISTS EPBLAS_PARALLEL_TESTABLE_TARGETS)
    set(LIB_PREFIX "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_overlay_archive "${LIB_PREFIX}blas_parallel")
    set(_migrated_archive "${LIB_PREFIX}blas_migrated")
    set(_composite "${LIB_PREFIX}blas")

    if(NOT TARGET ${_overlay_archive} OR NOT TARGET ${_migrated_archive})
        message(STATUS "test/blas_parallel: overlay or _migrated absent for "
                       "target '${_target}' — skipping")
        continue()
    endif()

    set(_common_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/common/target_${_target})
    if(NOT IS_DIRECTORY ${_common_target_dir})
        message(STATUS "test/blas_parallel: no fuzz_util shim for target "
                       "'${_target}' — skipping")
        continue()
    endif()

    set(_consist_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/consistency/target_${_target})
    set(_perf_target_dir
        ${PROJECT_SOURCE_DIR}/benchmark/drivers/target_${_target})

    # Per-target fuzz_util library.
    add_fypp_source(_fuzz_util_src
        ${_common_target_dir}/fuzz_util.fypp
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
    set(_fuzz_common ${LIB_PREFIX}blas_parallel_fuzz_common)
    add_library(${_fuzz_common} STATIC ${_fuzz_util_src})
    # Each fuzz_common library defines a module named `fuzz_util` —
    # they MUST land in separate dirs or the per-target .mod files
    # race to overwrite each other under the shared layout
    # `fortran_module_layout` would otherwise install.
    set(_fuzz_moddir ${CMAKE_CURRENT_BINARY_DIR}/fmod/${_target})
    set_target_properties(${_fuzz_common} PROPERTIES
        Fortran_MODULE_DIRECTORY ${_fuzz_moddir})
    target_include_directories(${_fuzz_common} PUBLIC ${_fuzz_moddir})
    if("${_target}" STREQUAL "multifloats" AND _have_multifloats_f)
        target_link_libraries(${_fuzz_common} PUBLIC
            $<BUILD_INTERFACE:multifloatsf>)
    endif()

    # Consistency tests — overlay vs renamed-migrated diff.
    if(IS_DIRECTORY ${_consist_target_dir})
        file(GLOB _consist_shims CONFIGURE_DEPENDS
            ${_consist_target_dir}/fuzz_*.fypp)
        foreach(_s ${_consist_shims})
            get_filename_component(_name ${_s} NAME_WE)
            add_fypp_source(_src ${_s}
                ${CMAKE_CURRENT_SOURCE_DIR}/consistency)
            add_executable(${_name} ${_src})
            target_link_libraries(${_name} PRIVATE
                ${_fuzz_common}
                ${_composite}
                ${_migrated_archive}
                eplinalg::blas)
            add_test(NAME epblas_parallel_${_name} COMMAND ${_name})
            set_tests_properties(epblas_parallel_${_name} PROPERTIES
                ENVIRONMENT "BLAS_FUZZ_CASES=200"
                LABELS "fuzz")
        endforeach()
    endif()

    # (The legacy Fortran bench harness was removed here; archived locally
    # under workspace/archive/tests/epblas-parallel/bench.)

    # C perf harnesses — kernel-isolated timing without the Fortran
    # bench's per-iter realloc/copy asymmetry. The
    # -ffunction-sections + --gc-sections combo keeps overlay and
    # migrated entry points close in the linked binary (avoids
    # ~10% iTLB/page-walk skew from the rest of the eyblas_parallel
    # archive landing between them).
    if(IS_DIRECTORY ${_perf_target_dir})
        file(GLOB _perf_srcs CONFIGURE_DEPENDS
            ${_perf_target_dir}/perf_*.c
            ${_perf_target_dir}/perf_*.cpp)
        foreach(_p ${_perf_srcs})
            get_filename_component(_name ${_p} NAME_WE)
            add_executable(${_name} ${_p})
            target_link_libraries(${_name} PRIVATE
                ${_composite}
                ${_migrated_archive}
                eplinalg::blas)
            # The multifloats migrated reference (mwblas) calls the
            # multifloats Fortran module (dd_mul_dd, …); fp80/quad
            # references are intrinsic and need no runtime. Mirror the
            # fuzz_common link so perf drivers resolve those symbols.
            if(TARGET multifloatsf AND "${_target}" STREQUAL "multifloats")
                target_link_libraries(${_name} PRIVATE
                    $<BUILD_INTERFACE:multifloatsf>)
            endif()
            target_compile_options(${_name} PRIVATE
                -ffunction-sections -fdata-sections)
            target_link_options(${_name} PRIVATE
                LINKER:--gc-sections)
            add_test(NAME epblas_parallel_${_name} COMMAND ${_name})
            set_tests_properties(epblas_parallel_${_name}
                PROPERTIES LABELS "perf")
        endforeach()
    endif()
endforeach()

# Wait-policy livelock regression for etrsm's OMP=4 threading.
#
# etrsm once livelocked ~16-23% of OMP=4 runs: a libgomp lost-wakeup race where
# a barrier wake fired during the *default* finite spin-then-block window gets
# lost, leaving a worker blocked forever. It was fixed structurally by the
# 03f48ac rewrite of etrsm_parallel.c — now a single flat non-nesting parallel
# region with NO explicit barrier (only the implicit join), an omp_in_parallel()
# guard forcing nested calls serial, and no `#pragma omp` in the kernel/pack/
# serial path, so the vulnerable blocked_dispatch dynamic barrier is gone.
#
# Policy note: the race is specific to the spin-then-block window. passive
# (block immediately) and active (spin forever, never block) are both immune;
# only the DEFAULT policy is exposed. So this guard must force that window —
# GOMP_SPINCOUNT=300000 with OMP_WAIT_POLICY unset, so an inherited passive/
# active can't mask it — and run the historical trigger (perf_etrsm, heavy
# 4-thread region churn) under it with a hard TIMEOUT: if a future change
# reintroduces a nested team or a dispatch barrier the run hangs and the TIMEOUT
# fails the suite instead of the wedge resurfacing silently in production.
# The wedge is probabilistic, so one run is a best-effort guard, not a certain
# catch. Verified clean on the current code: 0/30 perf runs + 60k threaded
# regions, no wedge. (memory project-etrsm-omp4-wedge)
if(TARGET perf_etrsm)
    add_test(NAME epblas_parallel_etrsm_wedge_guard COMMAND perf_etrsm)
    set_tests_properties(epblas_parallel_etrsm_wedge_guard PROPERTIES
        ENVIRONMENT_MODIFICATION "OMP_WAIT_POLICY=unset:;GOMP_SPINCOUNT=set:300000;OMP_NUM_THREADS=set:4;BLAS_PERF_ITERS=set:20;BLAS_PERF_TIME_BUDGET=set:0.1"
        TIMEOUT 90
        LABELS "fuzz")
endif()

# Source-level lint: enforce the "int is a boundary-only type" rule. The overlay
# uses ptrdiff_t (sizes/indices/counts), bool (logical flags), and char (BLAS
# option letters) internally; the fixed-width Fortran ABI types int/int64_t are
# confined to the two boundaries in common/ (epblas_facade.h facades + blas_omp.h
# OpenMP wrapper). The guard greps kind10/ kind16/ multifloats/ for any bare
# `int` (comments and string literals stripped) and fails the suite if one
# reappears. No built target needed — it is a pure source check.
find_program(EPBLAS_PARALLEL_PYTHON3 NAMES python3 python)
if(EPBLAS_PARALLEL_PYTHON3)
    add_test(NAME epblas_parallel_int_boundary_guard
        COMMAND ${EPBLAS_PARALLEL_PYTHON3}
                ${CMAKE_SOURCE_DIR}/test/check_int_boundary.py)
    set_tests_properties(epblas_parallel_int_boundary_guard PROPERTIES
        TIMEOUT 60
        LABELS "lint")
endif()
