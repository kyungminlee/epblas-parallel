# epblas-parallel — hand-written extended-precision BLAS overlays.
#
# Production build needs no `eplinalg` install: every target whose
# `src/epblas-parallel/<target>/` directory exists builds its overlay
# archive + INTERFACE composite. `eplinalg` is only a test/bench
# dependency, gated below before `add_subdirectory(tests)`.
#
# Repo ships two CMake packages:
#   * epblas-parallel — production parallel overlay + overlay-equipped
#                       composite (`epblas-parallel::{e,q,m}blas`)
#   * epblas-openblas — OpenBLAS-derived reference archive used for A/B
#                       comparisons (`epblas-openblas::{e,q,m}blas`)

cmake_minimum_required(VERSION 3.25)

project(epblas-parallel
    VERSION 0.1.0
    LANGUAGES C CXX Fortran)

# Fortran is enabled at project() time (not behind BUILD_TESTING) so the
# test suite's later find_package(<prefix>blas) calls — which derive a
# compiler tag from CMAKE_Fortran_COMPILER_ID — work uniformly. With
# Fortran inactive that tag is empty and the discovery probe silently
# fails.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FortranCompiler)

# Baked-in (target, prefix) table. Adding a fourth target is a
# coordinated change touching both repos; see CONTEXT.md §"Discovery".
set(EPBLAS_PARALLEL_KNOWN_TARGETS kind10 kind16 multifloats)
set(EPBLAS_PARALLEL_PREFIX_kind10       e)
set(EPBLAS_PARALLEL_PREFIX_kind16       q)
set(EPBLAS_PARALLEL_PREFIX_multifloats  m)

# Production discovery: a target builds iff its src dir exists. The
# eplinalg per-target package is only checked later, before the test
# subdirs (see EPBLAS_PARALLEL_TESTABLE_TARGETS below).
option(EPBLAS_PARALLEL_BUILD_multifloats
    "Build the multifloats overlay target (pulls multifloats via FetchContent)" ON)
set(EPBLAS_PARALLEL_BUILT_TARGETS "")
foreach(_target IN LISTS EPBLAS_PARALLEL_KNOWN_TARGETS)
    if("${_target}" STREQUAL "multifloats" AND NOT EPBLAS_PARALLEL_BUILD_multifloats)
        message(STATUS "epblas-parallel: multifloats target disabled via "
                       "-DEPBLAS_PARALLEL_BUILD_multifloats=OFF — skipping")
        continue()
    endif()
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/epblas-parallel/${_target})
        message(STATUS "epblas-parallel: building target ${_target}")
        list(APPEND EPBLAS_PARALLEL_BUILT_TARGETS ${_target})
    else()
        message(STATUS "epblas-parallel: no src/epblas-parallel/${_target}/ — skipping")
    endif()
endforeach()

if(NOT EPBLAS_PARALLEL_BUILT_TARGETS)
    message(WARNING
        "epblas-parallel: no per-target kernel sets present under "
        "src/epblas-parallel/ — configuring an empty build.")
    return()
endif()

# multifloats is the upstream double-double library; the parallel
# overlay for the multifloats target compiles its C++ kernels
# against the `multifloats` target. Fetched the same way the
# migrator did — same upstream, same tag floor.
#
# Wrapped in a function so the BUILD_TESTING / MULTIFLOATS_BUILD_BENCH
# overrides stay function-scoped — a plain directory-scope `set()`
# would shadow the cache value of BUILD_TESTING for the rest of this
# CMakeLists.txt and silently disable the overlay test suite even when
# the user passed -DBUILD_TESTING=ON.
function(_epblas_add_multifloats_subdir)
    set(BUILD_TESTING OFF)
    set(MULTIFLOATS_BUILD_BENCH OFF)
    if(DEFINED MULTIFLOATS_DIR)
        message(STATUS "epblas-parallel: using local multifloats: ${MULTIFLOATS_DIR}")
        add_subdirectory(${MULTIFLOATS_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/_mf EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        set(MULTIFLOATS_GIT_REPO "https://github.com/kyungminlee/multifloats.git"
            CACHE STRING "Git URL for the multifloats library")
        set(MULTIFLOATS_GIT_TAG "v0.6.0"
            CACHE STRING "Git tag/branch/commit for multifloats (>= v0.6.0)")
        message(STATUS "epblas-parallel: fetching multifloats from ${MULTIFLOATS_GIT_REPO} (${MULTIFLOATS_GIT_TAG})")
        FetchContent_Declare(multifloats_fetch
            GIT_REPOSITORY ${MULTIFLOATS_GIT_REPO}
            GIT_TAG        ${MULTIFLOATS_GIT_TAG}
        )
        FetchContent_Populate(multifloats_fetch)
        add_subdirectory(
            ${multifloats_fetch_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/_mf EXCLUDE_FROM_ALL)
    endif()
endfunction()

if("multifloats" IN_LIST EPBLAS_PARALLEL_BUILT_TARGETS)
    _epblas_add_multifloats_subdir()
endif()

add_subdirectory(src/epblas-parallel)
add_subdirectory(src/epblas-openblas)

# ── Composite INTERFACE targets (epblas-parallel package) ─────────
# epblas-parallel::{e,q,m}blas wraps the overlay archive in
# WHOLE_ARCHIVE so every Fortran-callable symbol the overlay defines
# lands in the consumer's binary. Without WHOLE_ARCHIVE a consumer
# that chains another BLAS archive could resolve the same symbol
# from the wrong source; with it, the overlay always wins for the
# routines it covers. No serial baseline behind — a routine the
# overlay does not cover surfaces as an honest link error.
foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_overlay "${_prefix}blas_parallel")
    set(_composite "${_prefix}blas")

    if(NOT TARGET ${_overlay})
        message(STATUS "composite: ${_overlay} target absent — skipping ${_composite} composite")
        continue()
    endif()

    add_library(${_composite} INTERFACE)
    add_library(epblas-parallel::${_composite} ALIAS ${_composite})

    # OpenMP runtime must land in every executable that links the
    # overlay. The dep on _overlay alone doesn't propagate cleanly
    # through the WHOLE_ARCHIVE generator expression below, so add
    # it directly on the composite's INTERFACE.
    if(OpenMP_C_FOUND)
        target_link_libraries(${_composite} INTERFACE OpenMP::OpenMP_C)
    endif()

    # Use the namespaced alias inside $<LINK_LIBRARY:...>. install(EXPORT)
    # does NOT introspect generator-expression bodies — it copies them
    # verbatim. The bare target name `${_overlay}` would survive into the
    # installed Targets file and downstream linkers would try `-lqblas_parallel`
    # against the system search path. The `epblas-parallel::` form resolves to
    # the IMPORTED target the export installs alongside the composite.
    target_link_libraries(${_composite} INTERFACE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,epblas-parallel::${_overlay}>")

    message(STATUS "composite: epblas-parallel::${_composite} = WHOLE_ARCHIVE(epblas-parallel::${_overlay})")
endforeach()

# ── Install + export ──────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Build two packages out of one configure:
#   epblas-parallel — composite INTERFACEs + raw overlay archives
#   epblas-openblas — raw openblas-derived reference archives
# Excludes the `_migrated` IMPORTED archives from both — they're
# build-tree-only (test consumption), never installed.

set(_epblas_parallel_install_targets "")
set(_epblas_openblas_install_targets "")
foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    foreach(_t ${_prefix}blas_parallel ${_prefix}blas)  # raw + composite
        if(TARGET ${_t})
            list(APPEND _epblas_parallel_install_targets ${_t})
        endif()
    endforeach()
    if(TARGET ${_prefix}blas_openblas)
        list(APPEND _epblas_openblas_install_targets ${_prefix}blas_openblas)
    endif()
endforeach()

if(_epblas_parallel_install_targets)
    install(TARGETS ${_epblas_parallel_install_targets}
        EXPORT epblas-parallelTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(EXPORT epblas-parallelTargets
        NAMESPACE epblas-parallel::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-parallel
    )

    # OpenMP components needed by the consumer. C is always required;
    # CXX is required only when the multifloats target is built —
    # multifloats's C++ kernels PUBLIC-link OpenMP::OpenMP_CXX, so its
    # IMPORTED target must exist on the consumer side.
    set(EPBLAS_PARALLEL_OPENMP_COMPONENTS "C")
    if("multifloats" IN_LIST EPBLAS_PARALLEL_BUILT_TARGETS)
        string(APPEND EPBLAS_PARALLEL_OPENMP_COMPONENTS " CXX")
    endif()

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/epblas-parallelConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-parallelConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-parallel
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-parallelConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-parallelConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-parallelConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-parallel
    )
endif()

if(_epblas_openblas_install_targets)
    install(TARGETS ${_epblas_openblas_install_targets}
        EXPORT epblas-openblasTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(EXPORT epblas-openblasTargets
        NAMESPACE epblas-openblas::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-openblas
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/epblas-openblasConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-openblasConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-openblas
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-openblasConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-openblasConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-openblasConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-openblas
    )
endif()

# Overlay test suite. Needs the fypp executable on PATH (the
# consistency / fuzz / bench drivers are .fypp-templated Fortran).
# Fortran itself is already enabled at the top of this file.
option(BUILD_TESTING "Build the overlay test suite (needs fypp + eplinalg)" ON)
if(BUILD_TESTING)
    enable_testing()

    # Tests + benches A/B against the migrated baseline, so each
    # built target needs its eplinalg per-target package. Targets
    # without one configure cleanly but contribute no tests.
    # `find_package(<prefix>blas)` transitively imports
    # `eplinalg::blas` (std-precision, carries lsame_/xerbla_) via
    # the per-target Config's find_dependency(eplinalgStdBlas), so
    # no separate probe is needed.
    set(EPBLAS_PARALLEL_TESTABLE_TARGETS "")
    foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
        set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
        set(_pkg "${_prefix}blas")
        find_package(${_pkg} QUIET)
        if(${_pkg}_FOUND)
            message(STATUS "epblas-parallel: tests enabled for target ${_target} (${_pkg} found)")
            list(APPEND EPBLAS_PARALLEL_TESTABLE_TARGETS ${_target})
        else()
            message(STATUS "epblas-parallel: ${_pkg} not on CMAKE_PREFIX_PATH — tests for ${_target} disabled")
        endif()
    endforeach()

    add_subdirectory(tests)
endif()
