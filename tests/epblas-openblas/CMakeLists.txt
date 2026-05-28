# epblas-openblas overlay test suite — multi-target driver.
#
# Per target in EPBLAS_PARALLEL_BUILT_TARGETS that has an epblas-openblas
# kernel set, we build consistency / bench / perf executables that
# compare the epblas-openblas overlay archive against the
# `<prefix>blas_openblas_migrated` renamed-migrated archive (also
# produced by src/epblas-openblas/).
#
# Body .fypp templates are REUSED from tests/epblas-parallel/ — same
# Fortran ABI, parameterized by overlay-archive symbol name only.
# Per-target shims live here:
#   common/target_<name>/fuzz_util.fypp
#   consistency/target_<name>/fuzz_<routine>.fypp
#   bench/target_<name>/bench_<routine>.fypp

set(_have_multifloats_f FALSE)
if(TARGET multifloatsf)
    set(_have_multifloats_f TRUE)
endif()

foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(LIB_PREFIX "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_overlay_archive "${LIB_PREFIX}blas_openblas")
    set(_migrated_archive "${LIB_PREFIX}blas_openblas_migrated")
    set(_overlay_routines
        "${EPBLAS_OPENBLAS_OVERLAY_ROUTINES_${_target}}")

    if(NOT TARGET ${_overlay_archive} OR NOT TARGET ${_migrated_archive})
        message(STATUS "tests/epblas-openblas: overlay or _migrated absent "
                       "for target '${_target}' — skipping")
        continue()
    endif()

    set(_common_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/common/target_${_target})
    if(NOT IS_DIRECTORY ${_common_target_dir})
        message(STATUS "tests/epblas-openblas: no fuzz_util shim for target "
                       "'${_target}' — skipping")
        continue()
    endif()

    set(_consist_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/consistency/target_${_target})
    set(_bench_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/bench/target_${_target})

    # Body .fypp templates live in tests/epblas-parallel/ (same Fortran
    # ABI, nothing references "parallel" specifically).
    set(_parallel_common_body
        ${CMAKE_CURRENT_SOURCE_DIR}/../epblas-parallel/common)
    set(_parallel_consist_body
        ${CMAKE_CURRENT_SOURCE_DIR}/../epblas-parallel/consistency)
    set(_parallel_bench_body
        ${CMAKE_CURRENT_SOURCE_DIR}/../epblas-parallel/bench)

    # Per-target fuzz_util library.
    add_fypp_source(_fuzz_util_src
        ${_common_target_dir}/fuzz_util.fypp
        ${_parallel_common_body})
    set(_fuzz_common ${LIB_PREFIX}blas_openblas_fuzz_common)
    add_library(${_fuzz_common} STATIC ${_fuzz_util_src})
    fortran_module_layout(${_fuzz_common})
    if("${_target}" STREQUAL "multifloats" AND _have_multifloats_f)
        target_link_libraries(${_fuzz_common} PUBLIC
            $<BUILD_INTERFACE:multifloatsf>)
    endif()

    # Consistency tests — overlay vs renamed-migrated diff.
    # No `blas` link: lsame_/xerbla_ come from _migrated (which is
    # the full serial archive with only overlay routines renamed).
    if(IS_DIRECTORY ${_consist_target_dir})
        file(GLOB _consist_shims CONFIGURE_DEPENDS
            ${_consist_target_dir}/fuzz_*.fypp)
        foreach(_s ${_consist_shims})
            get_filename_component(_name ${_s} NAME_WE)
            add_fypp_source(_src ${_s} ${_parallel_consist_body})
            add_executable(ep_${_name} ${_src})
            target_link_libraries(ep_${_name} PRIVATE
                ${_fuzz_common}
                ${_overlay_archive}
                ${_migrated_archive})
            add_test(NAME epblas_openblas_${_name} COMMAND ep_${_name})
            set_tests_properties(epblas_openblas_${_name} PROPERTIES
                ENVIRONMENT "BLAS_FUZZ_CASES=200"
                LABELS "epblas-openblas-fuzz")
        endforeach()
    endif()

    # Bench tests — registered twice (OMP=1, OMP=4).
    if(IS_DIRECTORY ${_bench_target_dir})
        file(GLOB _bench_shims CONFIGURE_DEPENDS
            ${_bench_target_dir}/bench_*.fypp)
        foreach(_s ${_bench_shims})
            get_filename_component(_name ${_s} NAME_WE)
            add_fypp_source(_src ${_s} ${_parallel_bench_body})
            add_executable(ep_${_name} ${_src})
            target_link_libraries(ep_${_name} PRIVATE
                ${_fuzz_common}
                ${_overlay_archive}
                ${_migrated_archive})
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/reports/epblas-openblas)
            foreach(_threads 1 4)
                set(_test_name epblas_openblas_${_name}_omp${_threads})
                set(_json
                    ${CMAKE_BINARY_DIR}/reports/epblas-openblas/${_name}_omp${_threads}.json)
                add_test(NAME ${_test_name} COMMAND ep_${_name}
                         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/reports/epblas-openblas)
                set_tests_properties(${_test_name} PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=${_threads};BLAS_BENCH_OUT=${_json}"
                    LABELS "epblas-openblas-bench;epblas-openblas-bench-omp${_threads}")
            endforeach()
        endforeach()
    endif()

    # C perf harnesses — reused from tests/epblas-parallel/perf/target_<T>/,
    # built only for routines that are in the epblas-openblas overlay set
    # so the link succeeds without dragging in unported symbols.
    set(_perf_target_dir
        ${CMAKE_CURRENT_SOURCE_DIR}/../epblas-parallel/perf/target_${_target})
    if(IS_DIRECTORY ${_perf_target_dir} AND _overlay_routines)
        foreach(_r ${_overlay_routines})
            set(_perf_src ${_perf_target_dir}/perf_${_r}.c)
            set(_have_src FALSE)
            if(EXISTS ${_perf_src})
                set(_have_src TRUE)
            else()
                set(_perf_src ${_perf_target_dir}/perf_${_r}.cpp)
                if(EXISTS ${_perf_src})
                    set(_have_src TRUE)
                endif()
            endif()
            if(NOT _have_src)
                continue()
            endif()
            get_filename_component(_name ${_perf_src} NAME_WE)
            add_executable(ep_${_name} ${_perf_src})
            target_link_libraries(ep_${_name} PRIVATE
                ${_overlay_archive}
                ${_migrated_archive})
            target_compile_options(ep_${_name} PRIVATE
                -ffunction-sections -fdata-sections)
            target_link_options(ep_${_name} PRIVATE LINKER:--gc-sections)
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/reports/epblas-openblas)
            foreach(_threads 1 4)
                set(_test_name epblas_openblas_${_name}_omp${_threads})
                set(_json
                    ${CMAKE_BINARY_DIR}/reports/epblas-openblas/${_name}_omp${_threads}.json)
                add_test(NAME ${_test_name} COMMAND ep_${_name}
                         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/reports/epblas-openblas)
                set_tests_properties(${_test_name} PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=${_threads};BLAS_PERF_JSON=${_json}"
                    LABELS "epblas-openblas-perf;epblas-openblas-perf-omp${_threads}")
            endforeach()
        endforeach()
    endif()
endforeach()
