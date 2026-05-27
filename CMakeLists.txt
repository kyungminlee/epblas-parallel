# epparablas — hand-written extended-precision BLAS overlays.
#
# Depends on `fortran-migrator` being installed; finds its per-target
# packages on CMAKE_PREFIX_PATH. Builds overlays + composites for
# every target whose migrator install is discoverable. See CONTEXT.md.

cmake_minimum_required(VERSION 3.25)

project(epparablas
    VERSION 0.1.0
    LANGUAGES C CXX)

# Baked-in (target, prefix) table. Adding a fourth target is a
# coordinated change touching both repos; see CONTEXT.md §"Discovery".
set(EPPARABLAS_KNOWN_TARGETS kind10 kind16 multifloats)
set(EPPARABLAS_PREFIX_kind10       e)
set(EPPARABLAS_PREFIX_kind16       q)
set(EPPARABLAS_PREFIX_multifloats  m)

set(EPPARABLAS_BUILT_TARGETS "")
foreach(_target IN LISTS EPPARABLAS_KNOWN_TARGETS)
    set(_prefix "${EPPARABLAS_PREFIX_${_target}}")
    set(_pkg "${_prefix}blas")
    find_package(${_pkg} QUIET)
    if(${_pkg}_FOUND)
        message(STATUS "epparablas: found ${_pkg} (target ${_target}) — overlay will build")
        list(APPEND EPPARABLAS_BUILT_TARGETS ${_target})
    else()
        message(STATUS "epparablas: ${_pkg} not found — skipping target ${_target}")
    endif()
endforeach()

if(NOT EPPARABLAS_BUILT_TARGETS)
    message(WARNING
        "epparablas: no fortran-migrator targets discovered on CMAKE_PREFIX_PATH "
        "— configuring an empty build. Install fortran-migrator for at least "
        "one of: ${EPPARABLAS_KNOWN_TARGETS}.")
    return()
endif()

# multifloats is the upstream double-double library; the parallel
# overlay for the multifloats target compiles its C++ kernels
# against the `multifloats` target. Fetched the same way the
# migrator did — same upstream, same tag floor.
if("multifloats" IN_LIST EPPARABLAS_BUILT_TARGETS)
    set(BUILD_TESTING OFF)
    set(MULTIFLOATS_BUILD_BENCH OFF)
    if(DEFINED MULTIFLOATS_DIR)
        message(STATUS "epparablas: using local multifloats: ${MULTIFLOATS_DIR}")
        add_subdirectory(${MULTIFLOATS_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/_mf EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        set(MULTIFLOATS_GIT_REPO "https://github.com/kyungminlee/multifloats.git"
            CACHE STRING "Git URL for the multifloats library")
        set(MULTIFLOATS_GIT_TAG "v0.6.0"
            CACHE STRING "Git tag/branch/commit for multifloats (>= v0.6.0)")
        message(STATUS "epparablas: fetching multifloats from ${MULTIFLOATS_GIT_REPO} (${MULTIFLOATS_GIT_TAG})")
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

add_subdirectory(src/parallel_blas)
add_subdirectory(src/epopenblas)

# ── Composite INTERFACE targets ───────────────────────────────────
# epparablas::{e,q,m}blas links the overlay archive *before* the
# serial migrated archive so overlay symbols shadow serial symbols
# at final link (gfortran name-mangling parity). WHOLE_ARCHIVE is
# mandatory: a static archive's members would otherwise only be
# pulled in to resolve unresolved externals — and since the serial
# archive already defines the same Fortran symbols, the overlay
# objects would be silently dropped.
foreach(_target IN LISTS EPPARABLAS_BUILT_TARGETS)
    set(_prefix "${EPPARABLAS_PREFIX_${_target}}")
    set(_overlay "${_prefix}blas_parallel")
    set(_serial  "epla::${_prefix}blas")
    set(_composite "${_prefix}blas")

    if(NOT TARGET ${_overlay})
        message(STATUS "composite: ${_overlay} target absent — skipping ${_composite} composite")
        continue()
    endif()

    add_library(${_composite} INTERFACE)
    add_library(epparablas::${_composite} ALIAS ${_composite})

    # OpenMP runtime must land in every executable that links the
    # overlay. The dep on _overlay alone doesn't propagate cleanly
    # through the WHOLE_ARCHIVE generator expression below, so add
    # it directly on the composite's INTERFACE.
    if(OpenMP_C_FOUND)
        target_link_libraries(${_composite} INTERFACE OpenMP::OpenMP_C)
    endif()

    target_link_libraries(${_composite} INTERFACE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_overlay}>")
    target_link_libraries(${_composite} INTERFACE ${_serial})

    message(STATUS "composite: epparablas::${_composite} = WHOLE_ARCHIVE(${_overlay}) → ${_serial}")
endforeach()

# ── Install + export ──────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Collect every installable target across the built-target set:
# overlay archives (parallel + epopenblas) + composite INTERFACEs.
# Excludes the `_migrated` IMPORTED archives — they're build-tree
# only (test consumption), never installed.
set(_epparablas_install_targets "")
foreach(_target IN LISTS EPPARABLAS_BUILT_TARGETS)
    set(_prefix "${EPPARABLAS_PREFIX_${_target}}")
    foreach(_t
            ${_prefix}blas_parallel
            ${_prefix}blas_epopenblas
            ${_prefix}blas)  # composite INTERFACE
        if(TARGET ${_t})
            list(APPEND _epparablas_install_targets ${_t})
        endif()
    endforeach()
endforeach()

if(_epparablas_install_targets)
    install(TARGETS ${_epparablas_install_targets}
        EXPORT epparablasTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(EXPORT epparablasTargets
        NAMESPACE epparablas::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epparablas
    )

    # Build the find_dependency() block for the migrator packages
    # this configure linked against. Composites import
    # `epla::<prefix>blas`, so each one needs the matching package
    # find_dependency-d when the consumer does find_package(epparablas).
    set(EPPARABLAS_FIND_DEPENDENCY_BLOCK "")
    foreach(_target IN LISTS EPPARABLAS_BUILT_TARGETS)
        set(_prefix "${EPPARABLAS_PREFIX_${_target}}")
        string(APPEND EPPARABLAS_FIND_DEPENDENCY_BLOCK
            "find_dependency(${_prefix}blas)\n")
    endforeach()

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/epparablasConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/epparablasConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epparablas
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/epparablasConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/epparablasConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/epparablasConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epparablas
    )
endif()

# Overlay test suite. Needs Fortran (the consistency / fuzz / bench
# drivers are .fypp-templated Fortran) and the fypp executable.
option(BUILD_TESTING "Build the overlay test suite (needs Fortran + fypp)" ON)
if(BUILD_TESTING)
    enable_language(Fortran)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(FortranCompiler)
    enable_testing()
    add_subdirectory(tests)
endif()

