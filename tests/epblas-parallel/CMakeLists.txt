# Parallel BLAS overlay test suite — multi-target driver.
#
# Per target in EPBLAS_PARALLEL_TESTABLE_TARGETS we generate:
#   common/target_<name>/fuzz_util.fypp        → fuzz_util.f90 (per-target shim)
#   consistency/target_<name>/fuzz_*.fypp      → fuzz drivers (overlay vs migrated)
#   bench/target_<name>/bench_*.fypp           → Fortran benches
#   perf/target_<name>/perf_*.c|.cpp           → C/C++ perf harnesses
#
# Each test binary links the composite `epblas-parallel::<prefix>blas`
# (WHOLE_ARCHIVE-wrapped overlay) AND the renamed `_migrated` archive
# (`<prefix>blas_migrated` — produced by tests/CMakeLists.txt) so a
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
        message(STATUS "tests/blas_parallel: overlay or _migrated absent for "
                       "target '${_target}' — skipping")
        continue()
    endif()

    set(_common_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/common/target_${_target})
    if(NOT IS_DIRECTORY ${_common_target_dir})
        message(STATUS "tests/blas_parallel: no fuzz_util shim for target "
                       "'${_target}' — skipping")
        continue()
    endif()

    set(_consist_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/consistency/target_${_target})
    set(_bench_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/bench/target_${_target})
    set(_perf_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/perf/target_${_target})

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

    # Bench tests.
    if(IS_DIRECTORY ${_bench_target_dir})
        file(GLOB _bench_shims CONFIGURE_DEPENDS
            ${_bench_target_dir}/bench_*.fypp)
        foreach(_s ${_bench_shims})
            get_filename_component(_name ${_s} NAME_WE)
            add_fypp_source(_src ${_s}
                ${CMAKE_CURRENT_SOURCE_DIR}/bench)
            add_executable(${_name} ${_src})
            target_link_libraries(${_name} PRIVATE
                ${_fuzz_common}
                ${_composite}
                ${_migrated_archive}
                eplinalg::blas)
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/reports)
            add_test(NAME epblas_parallel_${_name} COMMAND ${_name}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/reports)
            set_tests_properties(epblas_parallel_${_name}
                PROPERTIES LABELS "bench")
        endforeach()
    endif()

    # C perf harnesses — kernel-isolated timing without the Fortran
    # bench's per-iter realloc/copy asymmetry. The
    # -ffunction-sections + --gc-sections combo keeps overlay and
    # migrated entry points close in the linked binary (avoids
    # ~10% iTLB/page-walk skew from the rest of the eblas_parallel
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

# Wait-policy safety regression for etrsm's OMP=4 threading.
#
# etrsm once livelocked ~16% of OMP=4 runs under libgomp's *active* barrier
# spin (the nested-team wedge). The fix is structural and lives in
# etrsm_parallel.c: a single non-nesting parallel region with no explicit
# barrier, an omp_in_parallel() guard that forces nested calls serial, and no
# `#pragma omp` anywhere in the kernel/pack/serial path. The default fuzz tests
# run under whatever wait policy ctest inherits (typically passive), which would
# MASK a regression that reintroduces nesting or a barrier. This test pins the
# guarantee: run the etrsm fuzz under OMP_WAIT_POLICY=active + OMP=4 with a hard
# TIMEOUT, so a reintroduced wedge fails the suite (livelock -> timeout) instead
# of hanging silently in production under active. Verified clean: 36000 cases
# across 3 seeds pass under active, no hang. (memory project-etrsm-omp4-wedge)
#
# Case count is the full 12000-case bar (still ~1.5s under active): fuzz sizes
# are log-uniform in [1,128] biased to small matrices, and etrsm only threads at
# mnk >= 64^3, so a high count is needed to land enough genuine 4-thread regions
# (the ones the wedge lives in) each run.
if(TARGET fuzz_etrsm)
    add_test(NAME epblas_parallel_fuzz_etrsm_active COMMAND fuzz_etrsm)
    set_tests_properties(epblas_parallel_fuzz_etrsm_active PROPERTIES
        ENVIRONMENT "BLAS_FUZZ_CASES=12000;OMP_NUM_THREADS=4;OMP_WAIT_POLICY=active"
        TIMEOUT 120
        LABELS "fuzz")
endif()
