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
endif()

# Overlay subdirs, composite definitions, install/export land in later steps.
