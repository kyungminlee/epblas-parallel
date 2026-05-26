# epblas — hand-written extended-precision BLAS overlays.
#
# Depends on `fortran-migrator` being installed; finds its per-target
# packages on CMAKE_PREFIX_PATH. Builds overlays + composites for
# every target whose migrator install is discoverable. See CONTEXT.md.

cmake_minimum_required(VERSION 3.25)

project(epblas
    VERSION 0.1.0
    LANGUAGES C CXX)

# Baked-in (target, prefix) table. Adding a fourth target is a
# coordinated change touching both repos; see CONTEXT.md §"Discovery".
set(EPBLAS_KNOWN_TARGETS kind10 kind16 multifloats)
set(EPBLAS_PREFIX_kind10       e)
set(EPBLAS_PREFIX_kind16       q)
set(EPBLAS_PREFIX_multifloats  m)

set(EPBLAS_BUILT_TARGETS "")
foreach(_target IN LISTS EPBLAS_KNOWN_TARGETS)
    set(_prefix "${EPBLAS_PREFIX_${_target}}")
    set(_pkg "${_prefix}blas")
    find_package(${_pkg} QUIET)
    if(${_pkg}_FOUND)
        message(STATUS "epblas: found ${_pkg} (target ${_target}) — overlay will build")
        list(APPEND EPBLAS_BUILT_TARGETS ${_target})
    else()
        message(STATUS "epblas: ${_pkg} not found — skipping target ${_target}")
    endif()
endforeach()

if(NOT EPBLAS_BUILT_TARGETS)
    message(WARNING
        "epblas: no fortran-migrator targets discovered on CMAKE_PREFIX_PATH "
        "— configuring an empty build. Install fortran-migrator for at least "
        "one of: ${EPBLAS_KNOWN_TARGETS}.")
    return()
endif()

# multifloats is the upstream double-double library; the parallel
# overlay for the multifloats target compiles its C++ kernels
# against the `multifloats` target. Fetched the same way the
# migrator did — same upstream, same tag floor.
if("multifloats" IN_LIST EPBLAS_BUILT_TARGETS)
    set(BUILD_TESTING OFF)
    set(MULTIFLOATS_BUILD_BENCH OFF)
    if(DEFINED MULTIFLOATS_DIR)
        message(STATUS "epblas: using local multifloats: ${MULTIFLOATS_DIR}")
        add_subdirectory(${MULTIFLOATS_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/_mf EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        set(MULTIFLOATS_GIT_REPO "https://github.com/kyungminlee/multifloats.git"
            CACHE STRING "Git URL for the multifloats library")
        set(MULTIFLOATS_GIT_TAG "v0.6.0"
            CACHE STRING "Git tag/branch/commit for multifloats (>= v0.6.0)")
        message(STATUS "epblas: fetching multifloats from ${MULTIFLOATS_GIT_REPO} (${MULTIFLOATS_GIT_TAG})")
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

# Composite definitions and install/export land in later steps.
