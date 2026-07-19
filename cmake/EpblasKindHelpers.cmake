# EpblasKindHelpers.cmake — shared boilerplate for the six per-kind archive
# CMakeLists (src/epblas-parallel/{kind10,kind16,multifloats} and
# src/epblas-openblas/{kind10,kind16,multifloats}).
#
# Factors ONLY the blocks that were verbatim-identical (modulo trivially
# parameterizable names) across the six files: the routine-name suffix
# strip, add_library + namespaced alias, language standard, the
# -O3 + per-kind flag block, and the guarded -march append. Everything
# performance-load-bearing and genuinely per-kind stays in the per-kind
# files: the measured flag deltas (-fcx-fortran-rules / -falign-loops=16 /
# -ffp-contract={fast,on}) are passed through COMPILE_OPTIONS verbatim, and
# the per-file -falign-loops=32 set_source_files_properties pins, SIMD
# dispatch blocks, OpenMP component choice, and link libraries remain in
# each per-kind CMakeLists untouched.
#
# Included once from the top-level CMakeLists (functions are global).

# epblas_kernel_routines(<out_var> <kernel-src>...)
#
# Strip the source extension (.c / .cpp) from each kernel filename to get
# the Fortran-callable routine name it defines. Replaces the verbatim
# suffix-strip foreach that appeared in all six per-kind files. Split L3
# sources (*_parallel/*_serial/kernels) must NOT be passed here — their
# routine names are appended by hand at each call site.
function(epblas_kernel_routines out_var)
    set(_routines "")
    foreach(_k IN LISTS ARGN)
        string(REGEX REPLACE "\\.(c|cpp)$" "" _r "${_k}")
        list(APPEND _routines "${_r}")
    endforeach()
    set(${out_var} "${_routines}" PARENT_SCOPE)
endfunction()

# epblas_add_kind_archive(<target>
#     NAMESPACE  <epblas-parallel|epblas-openblas>
#     LANGUAGE   <C|CXX>          # C -> c_std_99, CXX -> cxx_std_17
#     MARCH      "<value>"        # -march=<value> appended last; "" disables
#     [EXPORT_NAME <name>]        # consumer-facing export name (openblas tree)
#     [COMMON_INCLUDE_DIR <dir>]  # PRIVATE include dir (openblas ../common)
#     SOURCES <src>...
#     COMPILE_OPTIONS <flag>...)  # per-kind flags, appended after -O3 in order
#
# Emits the STATIC archive exactly as the per-kind files previously spelled
# it: add_library + `<NAMESPACE>::<target>` alias, language standard, then
# `-O3 <COMPILE_OPTIONS...>` in one target_compile_options call and the
# -march flag in a second guarded call — so the effective flag order is
# byte-identical to the pre-dedup files.
#
# Runs in the caller's directory scope, so set_source_files_properties
# calls in the per-kind CMakeLists keep binding to this target's sources
# exactly as before.
function(epblas_add_kind_archive _target_name)
    cmake_parse_arguments(A ""
        "NAMESPACE;LANGUAGE;MARCH;EXPORT_NAME;COMMON_INCLUDE_DIR"
        "SOURCES;COMPILE_OPTIONS" ${ARGN})
    if(A_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "epblas_add_kind_archive(${_target_name}): "
                            "unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
    endif()

    add_library(${_target_name} STATIC ${A_SOURCES})
    add_library(${A_NAMESPACE}::${_target_name} ALIAS ${_target_name})

    if(A_EXPORT_NAME)
        set_target_properties(${_target_name} PROPERTIES
            EXPORT_NAME ${A_EXPORT_NAME})
    endif()
    if(A_COMMON_INCLUDE_DIR)
        target_include_directories(${_target_name} PRIVATE
            ${A_COMMON_INCLUDE_DIR})
    endif()

    if(A_LANGUAGE STREQUAL "CXX")
        target_compile_features(${_target_name} PRIVATE cxx_std_17)
    else()
        target_compile_features(${_target_name} PRIVATE c_std_99)
    endif()

    target_compile_options(${_target_name} PRIVATE -O3 ${A_COMPILE_OPTIONS})
    # -march comes from the tree-wide knob (EPBLAS_PARALLEL_MARCH /
    # EPBLAS_OPENBLAS_MARCH), passed in by value; empty disables.
    if(NOT "${A_MARCH}" STREQUAL "")
        target_compile_options(${_target_name} PRIVATE -march=${A_MARCH})
    endif()
endfunction()
