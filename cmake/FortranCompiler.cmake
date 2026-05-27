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
#     Sets up build-time and install-time module directories for the target.
#     Must be called before fortran_install_modules() or fortran_install_library().
#
#   fortran_install_modules(<target> [DESTINATION <base>])
#     Installs .mod/.smod files to <base>/fmod/<mod-tag>/.
#     Requires fortran_module_layout() to have been called on the target first.
#
#   fortran_install_library(<target> [NAMESPACE <ns>] [EXPORT <export-name>])
#     Installs the library with a compiler-tagged filename and generates a
#     Config.cmake for find_package() support.

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
# Helper: ensure INSTALL_INTERFACE paths are relative (required for
# relocatable packages). GNUInstallDirs can produce absolute paths on
# some platforms.
# ---------------------------------------------------------------------------
function(_fc_make_relative_installdir outvar path)
  if(IS_ABSOLUTE "${path}")
    file(RELATIVE_PATH _fc_relpath "${CMAKE_INSTALL_PREFIX}" "${path}")
    set(${outvar} "${_fc_relpath}" PARENT_SCOPE)
  else()
    set(${outvar} "${path}" PARENT_SCOPE)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# fortran_module_layout(<target>)
#
# Configures the target's module output directory and include paths.
# Uses FORTRAN_MOD_COMPAT_TAG for module directories.
# Must be called before fortran_install_modules() or fortran_install_library().
# ---------------------------------------------------------------------------
function(fortran_module_layout target)
  set(_moddir "${PROJECT_BINARY_DIR}/fmod/${FORTRAN_MOD_COMPAT_TAG}")

  set_target_properties(${target} PROPERTIES
    Fortran_MODULE_DIRECTORY "${_moddir}"
  )

  # Ensure the install interface path is relative for relocatable packages
  _fc_make_relative_installdir(_fc_rel_libdir "${CMAKE_INSTALL_LIBDIR}")

  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${_moddir}>
    $<INSTALL_INTERFACE:${_fc_rel_libdir}/fmod/${FORTRAN_MOD_COMPAT_TAG}>
  )
endfunction()

# ---------------------------------------------------------------------------
# fortran_install_modules(<target> [DESTINATION <base>])
#
# Installs .mod and .smod files to <base>/fmod/<mod-tag>/
# Default DESTINATION is ${CMAKE_INSTALL_LIBDIR}.
# Requires fortran_module_layout() to have been called on the target first.
# ---------------------------------------------------------------------------
function(fortran_install_modules target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "DESTINATION" "")
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "fortran_install_modules: unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # Validate that fortran_module_layout was called
  get_target_property(_fc_moddir ${target} Fortran_MODULE_DIRECTORY)
  if(NOT _fc_moddir)
    message(FATAL_ERROR
      "fortran_install_modules(${target}): Fortran_MODULE_DIRECTORY is not set. "
      "Call fortran_module_layout(${target}) first.")
  endif()

  if(NOT ARG_DESTINATION)
    set(ARG_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()

  set(_install_moddir "${ARG_DESTINATION}/fmod/${FORTRAN_MOD_COMPAT_TAG}")

  install(
    DIRECTORY "${_fc_moddir}/"
    DESTINATION "${_install_moddir}"
    FILES_MATCHING
      PATTERN "*.mod"
      PATTERN "*.smod"
  )
endfunction()

# ---------------------------------------------------------------------------
# fortran_install_library(<target>
#     [MPI]
#     [NAMESPACE <ns>]
#     [EXPORT <export-name>]
#     [DESTINATION <lib-dir>])
#
# Installs the library with a compiler-version-tagged filename (for ABI),
# while the export/config system uses the mod compat tag to find modules.
#
# When ``MPI`` is passed, the install ALSO tags the output with
# ``${MPI_TAG}`` (e.g. ``intelmpi-2021.18`` / ``openmpi-5.0`` /
# ``mpich-4.2``) so MPI-dependent libraries built against different
# implementations can coexist in the same install prefix. ``MPI_TAG``
# must be set by the caller (the top-level CMakeLists detects it from
# mpi.h's vendor macros). When ``MPI_TAG`` is unset, the MPI option is
# a no-op — the build falls back to compiler-only tagging.
#
# Note: This function generates a <ProjectName>Config.cmake. If your project
# has multiple Fortran library targets, add them all to a single EXPORT set
# by passing the same EXPORT name. The Config.cmake and export file are
# generated only once per export set.
# ---------------------------------------------------------------------------
function(fortran_install_library target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "MPI" "NAMESPACE;EXPORT;DESTINATION" "")
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "fortran_install_library: unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT ARG_NAMESPACE)
    set(ARG_NAMESPACE "${PROJECT_NAME}::")
  endif()
  if(NOT ARG_EXPORT)
    set(ARG_EXPORT "${PROJECT_NAME}Targets")
  endif()
  if(NOT ARG_DESTINATION)
    set(ARG_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()

  # Assemble the install tag. Compiler-version always; MPI flavor
  # appended for MPI-dependent libs when MPI_TAG is known.
  set(_full_tag "${FORTRAN_COMPILER_TAG}")
  set(_is_mpi_lib FALSE)
  if(ARG_MPI AND MPI_TAG)
    set(_full_tag "${FORTRAN_COMPILER_TAG}-${MPI_TAG}")
    set(_is_mpi_lib TRUE)
  endif()

  # Derive config name from the export set name (strip trailing "Targets").
  # This allows each library to get its own Config.cmake when given a
  # unique EXPORT name (e.g. EXPORT qblasTargets → qblasConfig.cmake).
  string(REGEX REPLACE "Targets$" "" _config_name "${ARG_EXPORT}")
  set(_cmake_install_dir "${ARG_DESTINATION}/cmake/${_config_name}")
  set(_targets_file "${ARG_EXPORT}-${_full_tag}.cmake")

  # Tag the library output filename by compiler + (optionally) MPI.
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${target}-${_full_tag}"
  )

  # Add target to the export set
  install(TARGETS ${target}
    EXPORT ${ARG_EXPORT}
    ARCHIVE DESTINATION "${ARG_DESTINATION}"
    LIBRARY DESTINATION "${ARG_DESTINATION}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )

  # Only generate the export file and Config.cmake once per export set.
  # Multiple targets can share the same EXPORT name; CMake accumulates them
  # from all install(TARGETS ... EXPORT <name>) calls. But install(EXPORT)
  # and Config.cmake generation must happen exactly once.
  get_property(_fc_config_generated GLOBAL PROPERTY _FC_CONFIG_GENERATED_${ARG_EXPORT})
  if(_fc_config_generated)
    return()
  endif()
  set_property(GLOBAL PROPERTY _FC_CONFIG_GENERATED_${ARG_EXPORT} TRUE)

  # Write the export set (includes all targets added to this EXPORT name)
  install(EXPORT ${ARG_EXPORT}
    FILE "${_targets_file}"
    NAMESPACE "${ARG_NAMESPACE}"
    DESTINATION "${_cmake_install_dir}"
  )

  # No cross-package sibling find_dependency() calls are emitted
  # into the generated Config.cmake — consumers list every archive
  # they need on their own link line; symbol resolution happens at
  # final link. See ../epparablas/CONTEXT.md §\"No sibling-deps
  # inside `epla` installed configs\" and adr/0001 for rationale.

  # Generate Config.cmake that finds the right targets file.
  # Strategy: derive the consumer's compiler tag (and MPI tag if this
  # is an MPI-dependent library) and look for an exact match. If no
  # exact match, fail with an informative error listing available builds.
  if(_is_mpi_lib)
    set(_mpi_detect_block "\
# --- MPI is a required dependency for this library ---
# Bring MPI::MPI_C / MPI::MPI_Fortran into scope so the Targets file's
# INTERFACE_LINK_LIBRARIES references resolve. The detection just
# below re-uses MPI_C_HEADER_DIR / the imported target's interface,
# both of which find_dependency(MPI) populates.
find_dependency(MPI COMPONENTS C Fortran)
# --- Derive consumer's MPI flavor + major.minor tag ---
find_package(MPI QUIET COMPONENTS C)
set(_FC_consumer_mpi_tag \"\")
if(MPI_C_FOUND)
  set(_FC_mpi_inc \"\${MPI_C_HEADER_DIR}\")
  if(TARGET MPI::MPI_C)
    get_target_property(_FC_mpi_iface MPI::MPI_C INTERFACE_INCLUDE_DIRECTORIES)
    if(_FC_mpi_iface)
      list(APPEND _FC_mpi_inc \${_FC_mpi_iface})
    endif()
  endif()
  list(REMOVE_DUPLICATES _FC_mpi_inc)
  set(_FC_mpi_probe \"\${CMAKE_CURRENT_BINARY_DIR}/_FC_mpi_probe.c\")
  file(WRITE \"\${_FC_mpi_probe}\" \"\\#include <mpi.h>\\n\\#include <stdio.h>\\nint main(void){\\n\\#if defined(I_MPI_VERSION)\\n  printf(\\\"intelmpi %s\\\\n\\\", I_MPI_VERSION);\\n\\#elif defined(OMPI_MAJOR_VERSION) && defined(OMPI_MINOR_VERSION)\\n  printf(\\\"openmpi %d.%d\\\\n\\\", OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION);\\n\\#elif defined(MPICH_VERSION)\\n  printf(\\\"mpich %s\\\\n\\\", MPICH_VERSION);\\n\\#else\\n  printf(\\\"unknown ?\\\\n\\\");\\n\\#endif\\n  return 0;\\n}\\n\")
  try_run(_FC_run_rc _FC_compile_rc
    \"\${CMAKE_CURRENT_BINARY_DIR}/_FC_mpi_probe.dir\"
    \"\${_FC_mpi_probe}\"
    CMAKE_FLAGS \"-DINCLUDE_DIRECTORIES=\${_FC_mpi_inc}\"
    RUN_OUTPUT_VARIABLE _FC_mpi_id)
  if(_FC_compile_rc AND \"\${_FC_run_rc}\" STREQUAL \"0\")
    string(STRIP \"\${_FC_mpi_id}\" _FC_mpi_id)
    if(_FC_mpi_id MATCHES \"^([a-z]+) ([0-9]+)\\\\.([0-9]+)\")
      set(_FC_consumer_mpi_tag \"\${CMAKE_MATCH_1}-\${CMAKE_MATCH_2}.\${CMAKE_MATCH_3}\")
    endif()
  endif()
  unset(_FC_mpi_inc)
  unset(_FC_mpi_iface)
  unset(_FC_mpi_probe)
  unset(_FC_mpi_id)
  unset(_FC_run_rc)
  unset(_FC_compile_rc)
endif()
if(NOT _FC_consumer_mpi_tag)
  set(\${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  set(\${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
    \"${_config_name}: MPI-dependent library but no MPI flavor detected on the consumer side. find_package(MPI) must succeed and mpi.h must be locatable.\")
  return()
endif()
")
    set(_consumer_tag_expr "\${_FC_consumer_tag}-\${_FC_consumer_mpi_tag}")
    set(_cleanup_mpi "unset(_FC_consumer_mpi_tag)")
  else()
    set(_mpi_detect_block "")
    set(_consumer_tag_expr "\${_FC_consumer_tag}")
    set(_cleanup_mpi "")
  endif()

  set(_config_content "\
# ${_config_name}Config.cmake
# Auto-generated by FortranCompiler.cmake
#
# Detects the consuming compiler and includes the matching targets file.
# Library files are tagged by compiler version (ABI compatibility).
# Module directories are tagged by .mod format version (compile-time compatibility).

cmake_minimum_required(VERSION 3.12)

# find_dependency() is used by the MPI block below for MPI-dependent
# libraries. No sibling find_dependency() calls are emitted any more
# (see ../epparablas/CONTEXT.md §\"No sibling-deps inside `epla`\").
include(CMakeFindDependencyMacro)

# --- Derive consumer's compiler family and ABI version tag ---
set(_FC_consumer_family \"\")

if(CMAKE_Fortran_COMPILER_ID STREQUAL \"GNU\")
  set(_FC_consumer_family \"gfortran\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"Intel\")
  set(_FC_consumer_family \"intel\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"IntelLLVM\")
  set(_FC_consumer_family \"intel\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"LLVMFlang\")
  set(_FC_consumer_family \"flang\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"Flang\")
  set(_FC_consumer_family \"flang-classic\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"NVHPC\")
  set(_FC_consumer_family \"nvhpc\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"NAG\")
  set(_FC_consumer_family \"nag\")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL \"Cray\")
  set(_FC_consumer_family \"cray\")
else()
  set(_FC_consumer_family \"\${CMAKE_Fortran_COMPILER_ID}\")
  string(TOLOWER \"\${_FC_consumer_family}\" _FC_consumer_family)
endif()

if(_FC_consumer_family STREQUAL \"gfortran\" OR _FC_consumer_family STREQUAL \"flang\")
  string(REGEX MATCH \"^([0-9]+)\" _FC_abi_version \"\${CMAKE_Fortran_COMPILER_VERSION}\")
elseif(_FC_consumer_family STREQUAL \"intel\")
  string(REGEX MATCH \"^([0-9]+\\\\.[0-9]+)\" _FC_abi_version \"\${CMAKE_Fortran_COMPILER_VERSION}\")
else()
  set(_FC_abi_version \"\${CMAKE_Fortran_COMPILER_VERSION}\")
endif()
set(_FC_consumer_tag \"\${_FC_consumer_family}-\${_FC_abi_version}\")
unset(_FC_abi_version)

${_mpi_detect_block}
# Look for exact (compiler[+MPI]) tag match
set(_FC_targets_file \"\${CMAKE_CURRENT_LIST_DIR}/${ARG_EXPORT}-${_consumer_tag_expr}.cmake\")

if(NOT EXISTS \"\${_FC_targets_file}\")
  # No exact match — list available builds and fail
  file(GLOB _FC_available \"\${CMAKE_CURRENT_LIST_DIR}/${ARG_EXPORT}-*.cmake\")
  set(_FC_available_names \"\")
  foreach(_FC_f IN LISTS _FC_available)
    get_filename_component(_FC_fname \"\${_FC_f}\" NAME)
    list(APPEND _FC_available_names \"\${_FC_fname}\")
  endforeach()
  list(JOIN _FC_available_names \", \" _FC_available_list)
  set(\${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  set(\${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
    \"${_config_name}: no pre-built library found for tag '${_consumer_tag_expr}'. Available: [\${_FC_available_list}]\")
  unset(_FC_consumer_family)
  unset(_FC_consumer_tag)
  unset(_FC_targets_file)
  unset(_FC_available)
  unset(_FC_available_names)
  unset(_FC_available_list)
  ${_cleanup_mpi}
  return()
endif()

include(\"\${_FC_targets_file}\")

unset(_FC_consumer_family)
unset(_FC_consumer_tag)
unset(_FC_targets_file)
${_cleanup_mpi}
")

  file(GENERATE
    OUTPUT "${PROJECT_BINARY_DIR}/cmake/${_config_name}Config.cmake"
    CONTENT "${_config_content}"
  )

  install(
    FILES "${PROJECT_BINARY_DIR}/cmake/${_config_name}Config.cmake"
    DESTINATION "${_cmake_install_dir}"
  )
endfunction()
