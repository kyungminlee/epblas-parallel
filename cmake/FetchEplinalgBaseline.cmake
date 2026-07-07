# FetchEplinalgBaseline.cmake
#
# The test/bench suite A/B's the parallel overlay against the migrated
# `eplinalg` baseline (`eyblas`/`qxblas`/`mwblas` + `eplinalgStdBlas`). Rather than
# require a hand-staged install on CMAKE_PREFIX_PATH, this module downloads the
# matching prebuilt release tarball for each built precision and extracts it
# under the build tree (${CMAKE_BINARY_DIR}/_deps/eplinalg-<p>), then prepends
# those prefixes to CMAKE_PREFIX_PATH so the find_package(<prefix>blas) probes
# below resolve. Downloads are cached in _deps/eplinalg-dl and extraction is
# idempotent, so reconfigures don't re-fetch.
#
# The baseline packages key their CMake Targets file on the EXACT consumer
# compiler tag (gfortran-<major>); releases ship gfortran-12 and gfortran-15,
# so the consuming build must use one of those Fortran compilers. A 16/unknown
# compiler simply skips the fetch and falls back to CMAKE_PREFIX_PATH.
#
# Knobs:
#   EPBLAS_PARALLEL_FETCH_BASELINE   ON  — fetch release baselines into the build
#   EPLINALG_BASELINE_VERSION        release tag to pull (default v0.8.0)
#   EPLINALG_BASELINE_MPI            MPI flavour in the asset name (default mpich)
#   EPLINALG_BASELINE_REPO           release host repo URL

option(EPBLAS_PARALLEL_FETCH_BASELINE
    "Download the eplinalg migrated baseline (eyblas/qxblas/mwblas) release \
binaries into the build tree for the test/bench suite" ON)
set(EPLINALG_BASELINE_VERSION "v0.8.0" CACHE STRING
    "eplinalg release tag for the prebuilt migrated baseline")
set(EPLINALG_BASELINE_MPI "mpich" CACHE STRING
    "MPI flavour tag in the eplinalg baseline release asset names")
set(EPLINALG_BASELINE_REPO "https://github.com/kyungminlee/eplinalg" CACHE STRING
    "Repo hosting the eplinalg baseline releases")

# Download + extract one precision's baseline tarball; return its prefix dir.
function(_epblas_fetch_one_baseline _precision _out_prefix)
    set(_asset "${_precision}-${_FC_baseline_tag}-${EPLINALG_BASELINE_MPI}-linux-x86_64.tar.gz")
    set(_url "${EPLINALG_BASELINE_REPO}/releases/download/${EPLINALG_BASELINE_VERSION}/${_asset}")
    set(_dl_dir "${CMAKE_BINARY_DIR}/_deps/eplinalg-dl")
    set(_tarball "${_dl_dir}/${_asset}")
    set(_prefix "${CMAKE_BINARY_DIR}/_deps/eplinalg-${_precision}")
    set(_stamp "${_prefix}/.eplinalg-baseline-version")

    # The asset name carries no release tag, so a cached tarball/extract from
    # a different EPLINALG_BASELINE_VERSION must not be reused — stamp the
    # extracted prefix with its version and purge both on mismatch.
    set(_have_version "")
    if(EXISTS "${_stamp}")
        file(READ "${_stamp}" _have_version)
    endif()
    if(NOT _have_version STREQUAL "${EPLINALG_BASELINE_VERSION}")
        file(REMOVE "${_tarball}")
        file(REMOVE_RECURSE "${_prefix}")
    endif()

    if(NOT EXISTS "${_tarball}")
        file(MAKE_DIRECTORY "${_dl_dir}")
        message(STATUS "epblas-parallel: downloading baseline ${_asset}")
        file(DOWNLOAD "${_url}" "${_tarball}" STATUS _st TLS_VERIFY ON)
        list(GET _st 0 _code)
        if(NOT _code EQUAL 0)
            list(GET _st 1 _msg)
            file(REMOVE "${_tarball}")
            message(FATAL_ERROR
                "epblas-parallel: failed to download ${_url}: ${_msg}\n"
                "  (no '${_FC_baseline_tag}' baseline for ${_precision}? releases ship "
                "gfortran-12/gfortran-15 — build with -DCMAKE_Fortran_COMPILER=gfortran-15, "
                "or set -DEPBLAS_PARALLEL_FETCH_BASELINE=OFF and provide CMAKE_PREFIX_PATH.)")
        endif()
    endif()

    # Idempotent extract: the per-precision blas Config is the sentinel.
    file(GLOB _have_cfg "${_prefix}/lib/cmake/*blas/*Config.cmake")
    if(NOT _have_cfg)
        file(MAKE_DIRECTORY "${_prefix}")
        file(ARCHIVE_EXTRACT INPUT "${_tarball}" DESTINATION "${_prefix}")
    endif()
    file(WRITE "${_stamp}" "${EPLINALG_BASELINE_VERSION}")
    set(${_out_prefix} "${_prefix}" PARENT_SCOPE)
endfunction()

function(epblas_fetch_eplinalg_baseline)
    if(NOT EPBLAS_PARALLEL_FETCH_BASELINE)
        return()
    endif()
    if(NOT CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        message(STATUS "epblas-parallel: FETCH_BASELINE skipped (Fortran compiler "
                       "'${CMAKE_Fortran_COMPILER_ID}' has no release baseline; "
                       "relying on CMAKE_PREFIX_PATH)")
        return()
    endif()
    string(REGEX MATCH "^([0-9]+)" _major "${CMAKE_Fortran_COMPILER_VERSION}")
    set(_FC_baseline_tag "gfortran-${_major}")

    set(_added "")
    foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
        # The release asset precision name is exactly the target name.
        _epblas_fetch_one_baseline("${_target}" _prefix)
        list(PREPEND _added "${_prefix}")
    endforeach()

    # Prepend into the cache var so the find_package() probes below see it.
    list(PREPEND CMAKE_PREFIX_PATH ${_added})
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    message(STATUS "epblas-parallel: baseline prefixes added: ${_added}")
endfunction()
