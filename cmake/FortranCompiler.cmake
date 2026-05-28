# FortranCompiler.cmake
#
# Detects the Fortran compiler's .mod file format version and provides
# functions for compiler-aware installation of Fortran modules and libraries.
#
# After include(FortranCompiler), the following variables are set:
#
#   FORTRAN_COMPILER_FAMILY   - Normalized compiler family (gfortran, intel, flang, nvhpc, nag, cray)
#   FORTRAN_COMPILER_VERSION  - Full compiler version string
#   FORTRAN_MOD_VERSION       - Internal .mod format version integer (or "unknown")
#   FORTRAN_MOD_COMPAT_TAG    - Module compat tag for .mod dirs (e.g. gfortran-mod15)
#   FORTRAN_COMPILER_TAG      - Compiler version tag for libraries (e.g. gfortran-14)
#
# Functions provided:
#
#   fortran_module_layout(<target>)
#     Sets up build-time module directories tagged by FORTRAN_MOD_COMPAT_TAG.
#
# NOTE: this file is a trimmed copy of the version in the sibling
# `eplinalg` repo. eplinalg also exports a compiler-version-tagged
# install layout via `fortran_install_modules` / `fortran_install_library`
# — those are not used by epblas-parallel and have been removed here.
# Keep this file in sync with eplinalg when the detection logic
# changes.

if(_FORTRAN_COMPILER_INCLUDED)
  return()
endif()
set(_FORTRAN_COMPILER_INCLUDED TRUE)

# ---------------------------------------------------------------------------
# Verify Fortran is enabled
# ---------------------------------------------------------------------------
get_property(_fc_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT "Fortran" IN_LIST _fc_languages)
  message(FATAL_ERROR "FortranCompiler: Fortran language must be enabled before including this module.")
endif()
unset(_fc_languages)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---------------------------------------------------------------------------
# Detect compiler family
#
# Compiler IDs used by CMake:
#   GNU       - gfortran
#   Intel     - ifort (classic)
#   IntelLLVM - ifx
#   LLVMFlang - LLVM Flang (flang-new, the official LLVM Fortran compiler)
#   Flang     - Classic Flang (PGI-derived, incompatible with LLVM Flang)
#   NVHPC     - NVIDIA nvfortran (PGI lineage, shares classic Flang .mod format)
#   NAG       - NAG Fortran
#   Cray      - Cray Fortran (CCE)
# ---------------------------------------------------------------------------
set(FORTRAN_COMPILER_VERSION "${CMAKE_Fortran_COMPILER_VERSION}")

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(FORTRAN_COMPILER_FAMILY "gfortran")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(FORTRAN_COMPILER_FAMILY "intel")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  set(FORTRAN_COMPILER_FAMILY "intel")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
  # LLVM Flang (flang-new): .mod files are valid Fortran source with !mod$ v1 header
  set(FORTRAN_COMPILER_FAMILY "flang")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Flang")
  # Classic Flang (PGI-derived): completely different .mod format from LLVM Flang
  set(FORTRAN_COMPILER_FAMILY "flang-classic")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  # NVIDIA nvfortran: shares PGI-lineage .mod format with classic Flang
  set(FORTRAN_COMPILER_FAMILY "nvhpc")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NAG")
  set(FORTRAN_COMPILER_FAMILY "nag")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  set(FORTRAN_COMPILER_FAMILY "cray")
else()
  set(FORTRAN_COMPILER_FAMILY "${CMAKE_Fortran_COMPILER_ID}")
  string(TOLOWER "${FORTRAN_COMPILER_FAMILY}" FORTRAN_COMPILER_FAMILY)
endif()

# ---------------------------------------------------------------------------
# Determine .mod format version from compiler family + version
# ---------------------------------------------------------------------------
set(FORTRAN_MOD_VERSION "unknown")

if(FORTRAN_COMPILER_FAMILY STREQUAL "gfortran")
  # GCC major version -> MOD_VERSION
  # GCC < 4.4: unversioned .mod format (unsupported)
  string(REGEX MATCH "^([0-9]+)" _fc_gcc_major "${FORTRAN_COMPILER_VERSION}")
  if(_fc_gcc_major VERSION_GREATER_EQUAL 15)
    set(FORTRAN_MOD_VERSION "16")
  elseif(_fc_gcc_major VERSION_GREATER_EQUAL 8)
    set(FORTRAN_MOD_VERSION "15")
  elseif(_fc_gcc_major VERSION_GREATER_EQUAL 5)
    set(FORTRAN_MOD_VERSION "14")
  elseif(_fc_gcc_major EQUAL 4)
    string(REGEX MATCH "^4\\.([0-9]+)" _fc_gcc_4minor "${FORTRAN_COMPILER_VERSION}")
    set(_fc_minor "${CMAKE_MATCH_1}")
    if(_fc_minor EQUAL 9)
      set(FORTRAN_MOD_VERSION "12")
    elseif(_fc_minor EQUAL 8)
      set(FORTRAN_MOD_VERSION "10")
    elseif(_fc_minor EQUAL 7)
      set(FORTRAN_MOD_VERSION "9")
    elseif(_fc_minor EQUAL 6)
      set(FORTRAN_MOD_VERSION "6")
    elseif(_fc_minor EQUAL 5)
      set(FORTRAN_MOD_VERSION "4")
    elseif(_fc_minor EQUAL 4)
      set(FORTRAN_MOD_VERSION "0")
    endif()
    unset(_fc_minor)
    unset(_fc_gcc_4minor)
  endif()
  unset(_fc_gcc_major)

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  # Classic ifort versioning
  # Version numbering jumped from 19.x to 2021.x (year-based oneAPI scheme).
  # All versions 18.x through 2021.x compare correctly with VERSION_GREATER_EQUAL
  # because 18 < 2020 < 2021 numerically.
  if(FORTRAN_COMPILER_VERSION VERSION_GREATER_EQUAL "2021.10")
    set(FORTRAN_MOD_VERSION "13")
  elseif(FORTRAN_COMPILER_VERSION VERSION_GREATER_EQUAL "18.0")
    set(FORTRAN_MOD_VERSION "12")
  elseif(FORTRAN_COMPILER_VERSION VERSION_GREATER_EQUAL "17.0")
    set(FORTRAN_MOD_VERSION "11")
  elseif(FORTRAN_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0")
    set(FORTRAN_MOD_VERSION "10")
  endif()

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  # ifx versioning (also uses year-based oneAPI scheme)
  if(FORTRAN_COMPILER_VERSION VERSION_GREATER_EQUAL "2023.2")
    set(FORTRAN_MOD_VERSION "13")
  elseif(FORTRAN_COMPILER_VERSION VERSION_GREATER_EQUAL "2021.0")
    set(FORTRAN_MOD_VERSION "12")
  endif()

elseif(FORTRAN_COMPILER_FAMILY STREQUAL "flang")
  # LLVM Flang: text-based .mod files with !mod$ v1 header
  set(FORTRAN_MOD_VERSION "1")

# flang-classic, nvhpc, nag, cray: no known .mod version mapping.
# They fall through to FORTRAN_MOD_VERSION = "unknown" and get tagged
# by full compiler version (conservative but safe).
endif()

# ---------------------------------------------------------------------------
# Build tags:
#   FORTRAN_MOD_COMPAT_TAG  - for .mod directories (by format compatibility)
#   FORTRAN_COMPILER_TAG    - for library files (by ABI-relevant version)
#
# Version truncation per family:
#   gfortran -> major only (ABI stable within a release series)
#   flang    -> major only (follows LLVM major versioning)
#   intel    -> major.minor (ABI can change at minor releases, e.g. 2021.10)
#   others   -> full version (conservative)
# ---------------------------------------------------------------------------
if(FORTRAN_COMPILER_FAMILY STREQUAL "gfortran" OR FORTRAN_COMPILER_FAMILY STREQUAL "flang")
  string(REGEX MATCH "^([0-9]+)" _fc_abi_version "${FORTRAN_COMPILER_VERSION}")
elseif(FORTRAN_COMPILER_FAMILY STREQUAL "intel")
  string(REGEX MATCH "^([0-9]+\\.[0-9]+)" _fc_abi_version "${FORTRAN_COMPILER_VERSION}")
else()
  set(_fc_abi_version "${FORTRAN_COMPILER_VERSION}")
endif()
set(FORTRAN_COMPILER_TAG "${FORTRAN_COMPILER_FAMILY}-${_fc_abi_version}")
unset(_fc_abi_version)

if(FORTRAN_MOD_VERSION STREQUAL "unknown")
  set(FORTRAN_MOD_COMPAT_TAG "${FORTRAN_COMPILER_TAG}")
else()
  set(FORTRAN_MOD_COMPAT_TAG "${FORTRAN_COMPILER_FAMILY}-mod${FORTRAN_MOD_VERSION}")
endif()

message(STATUS "FortranCompiler: compiler=${CMAKE_Fortran_COMPILER_ID} ${FORTRAN_COMPILER_VERSION}")
message(STATUS "FortranCompiler: family=${FORTRAN_COMPILER_FAMILY}, mod_version=${FORTRAN_MOD_VERSION}")
message(STATUS "FortranCompiler: mod_tag=${FORTRAN_MOD_COMPAT_TAG}, lib_tag=${FORTRAN_COMPILER_TAG}")

# ---------------------------------------------------------------------------
# fortran_module_layout(<target>)
#
# Configures the target's module output directory tagged by .mod
# format version. epblas-parallel never installs Fortran modules, so
# only the build-tree side is wired here.
# ---------------------------------------------------------------------------
function(fortran_module_layout target)
  set(_moddir "${PROJECT_BINARY_DIR}/fmod/${FORTRAN_MOD_COMPAT_TAG}")
  set_target_properties(${target} PROPERTIES
    Fortran_MODULE_DIRECTORY "${_moddir}"
  )
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${_moddir}>
  )
endfunction()

