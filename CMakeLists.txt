# epblas-parallel — hand-written extended-precision BLAS overlays.
#
# Depends on `eplinalg` being installed; finds its per-target packages
# on CMAKE_PREFIX_PATH. Builds overlays + composites for every target
# whose eplinalg install is discoverable. See CONTEXT.md.
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

# Fortran is enabled at project() time (not behind BUILD_TESTING) because
# the migrator's per-target Config.cmake files (eblasConfig.cmake, ...)
# derive a compiler tag from CMAKE_Fortran_COMPILER_ID at find_package
# time — with Fortran inactive that tag is empty, every find_package
# silently fails, and discovery reports zero targets.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FortranCompiler)

# Baked-in (target, prefix) table. Adding a fourth target is a
# coordinated change touching both repos; see CONTEXT.md §"Discovery".
set(EPBLAS_PARALLEL_KNOWN_TARGETS kind10 kind16 multifloats)
set(EPBLAS_PARALLEL_PREFIX_kind10       e)
set(EPBLAS_PARALLEL_PREFIX_kind16       q)
set(EPBLAS_PARALLEL_PREFIX_multifloats  m)

set(EPBLAS_PARALLEL_BUILT_TARGETS "")
foreach(_target IN LISTS EPBLAS_PARALLEL_KNOWN_TARGETS)
    set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_pkg "${_prefix}blas")
    find_package(${_pkg} QUIET)
    if(${_pkg}_FOUND)
        message(STATUS "epblas-parallel: found ${_pkg} (target ${_target}) — overlay will build")
        list(APPEND EPBLAS_PARALLEL_BUILT_TARGETS ${_target})
    else()
        message(STATUS "epblas-parallel: ${_pkg} not found — skipping target ${_target}")
    endif()
endforeach()

if(NOT EPBLAS_PARALLEL_BUILT_TARGETS)
    message(WARNING
        "epblas-parallel: no eplinalg targets discovered on CMAKE_PREFIX_PATH "
        "— configuring an empty build. Install eplinalg for at least "
        "one of: ${EPBLAS_PARALLEL_KNOWN_TARGETS}.")
    return()
endif()

# multifloats is the upstream double-double library; the parallel
# overlay for the multifloats target compiles its C++ kernels
# against the `multifloats` target. Fetched the same way the
# migrator did — same upstream, same tag floor.
if("multifloats" IN_LIST EPBLAS_PARALLEL_BUILT_TARGETS)
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
endif()

add_subdirectory(src/epblas-parallel)
add_subdirectory(src/epblas-openblas)

# ── Composite INTERFACE targets (epblas-parallel package) ─────────
# epblas-parallel::{e,q,m}blas links the overlay archive *before* the
# serial migrated archive so overlay symbols shadow serial symbols
# at final link (gfortran name-mangling parity). WHOLE_ARCHIVE is
# mandatory: a static archive's members would otherwise only be
# pulled in to resolve unresolved externals — and since the serial
# archive already defines the same Fortran symbols, the overlay
# objects would be silently dropped.
foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    set(_overlay "${_prefix}blas_parallel")
    set(_serial  "eplinalg::${_prefix}blas")
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
    target_link_libraries(${_composite} INTERFACE ${_serial})

    message(STATUS "composite: epblas-parallel::${_composite} = WHOLE_ARCHIVE(epblas-parallel::${_overlay}) → ${_serial}")
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

    # Build the find_dependency() block for the migrator packages
    # this configure linked against. Composites import
    # `eplinalg::<prefix>blas`, so each one needs the matching package
    # find_dependency-d when the consumer does find_package(epblas-parallel).
    set(EPBLAS_PARALLEL_FIND_DEPENDENCY_BLOCK "")
    foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
        set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
        string(APPEND EPBLAS_PARALLEL_FIND_DEPENDENCY_BLOCK
            "find_dependency(${_prefix}blas)\n")
    endforeach()

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
option(BUILD_TESTING "Build the overlay test suite (needs fypp)" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
