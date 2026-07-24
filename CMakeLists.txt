# epblas-parallel — hand-written extended-precision BLAS overlays.
#
# Production build needs no `eplinalg` install: every target whose
# `src/epblas-parallel/<target>/` directory exists builds its overlay
# archive + INTERFACE composite. `eplinalg` is only a test/bench
# dependency, gated below before `add_subdirectory(test)`.
#
# Repo ships two CMake packages:
#   * epblas-parallel — production parallel overlay + overlay-equipped
#                       composite (`epblas-parallel::{ey,qx,mw}blas`)
#   * epblas-openblas — OpenBLAS-derived reference archive used for A/B
#                       comparisons (`epblas-openblas::{ey,qx,mw}blas`)

cmake_minimum_required(VERSION 3.25)

# Single source of truth for the version: the top-level VERSION file.
# Bumping a release is a one-line edit there (see doc/dev/release.md); CMake,
# the installed package config, and include/epblas-parallel/version.h.in all
# read from PROJECT_VERSION so they never drift.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" EPBLAS_PARALLEL_VERSION
    LIMIT_COUNT 1 REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+")

project(epblas-parallel
    VERSION ${EPBLAS_PARALLEL_VERSION}
    LANGUAGES C CXX Fortran)

# Fortran is enabled at project() time (not behind BUILD_TESTING) so the
# test suite's later find_package(<prefix>blas) calls — which derive a
# compiler tag from CMAKE_Fortran_COMPILER_ID — work uniformly. With
# Fortran inactive that tag is empty and the discovery probe silently
# fails.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "epblas-parallel: Fortran compiler = "
               "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")

# Shared helpers for the six per-kind archive CMakeLists
# (epblas_add_kind_archive / epblas_kernel_routines).
include(EpblasKindHelpers)

# ── Position-independent code / shared-library output ─────────────────
# The overlay + reference archives are STATIC. Compiling them PIC is what
# lets a consumer wrap the archive in `--whole-archive` and link it into a
# shared object. The one thing that otherwise blocks that link is the
# persistent per-thread pack arenas (`static __thread g_pack`, from the
# pack-arena rollout): without PIC the compiler picks the local-exec TLS
# model (R_X86_64_TPOFF32), legal only inside an executable, so `-shared`
# fails with "local-exec is incompatible with -shared". PIC moves those
# file-local TLS vars to the local-dynamic model, which is shared-object-
# and dlopen-safe.
#
# Kept ON by default at negligible cost: on x86-64 the small code model
# already emits RIP-relative addressing (the static archives carry zero
# absolute R_X86_64_32/32S relocations either way), so hot-loop codegen is
# unchanged, and the pack-arena TLS handle is fetched O(1) per BLAS call —
# never per element. Set -DEPBLAS_POSITION_INDEPENDENT_CODE=OFF for a
# strictly executable-only static build.
option(EPBLAS_POSITION_INDEPENDENT_CODE
    "Compile overlay/reference archives position-independent (-fPIC)" ON)
if(EPBLAS_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Optional shared (.so) overlay libraries. Each is built by wrapping the
# STATIC overlay archive in `--whole-archive` so every Fortran-callable
# symbol is exported (Linux default visibility). Requires PIC — forced ON
# below when enabled. OFF by default so the installed/benchmarked
# artifacts remain the tuned static archives.
option(EPBLAS_BUILD_SHARED
    "Also build shared (.so) overlay libraries from the static archives" OFF)
if(EPBLAS_BUILD_SHARED AND NOT CMAKE_POSITION_INDEPENDENT_CODE)
    message(STATUS "epblas: EPBLAS_BUILD_SHARED=ON forces position-independent code")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Baked-in (target, prefix) table. Adding a fourth target is a
# coordinated change touching both repos.
#
# PREFIX (the real+complex routine-letter pair) names our CMake targets
# AND archive files, following eplinalg's v0.8.0 convention where
# package, target, and file name agree (`eyblas`, `eplinalg::eyblas`,
# `libeyblas-gfortran-15.a`). The eplinalg baseline pin (>= v0.8.0)
# uses the same two-letter names, so the one PREFIX table also names
# the baseline packages/targets we consume (`find_package(eyblas)`,
# `eplinalg::eyblas`).
set(EPBLAS_PARALLEL_KNOWN_TARGETS kind10 kind16 multifloats)
set(EPBLAS_PARALLEL_PREFIX_kind10       ey)
set(EPBLAS_PARALLEL_PREFIX_kind16       qx)
set(EPBLAS_PARALLEL_PREFIX_multifloats  mw)

# Per-target eplinalg baseline helper packages. Some per-precision BLAS
# baselines pull in precision-specific helper libs whose imported targets
# their own Config does NOT find_dependency — siblings are the consumer's
# responsibility (see eplinalg CONTEXT §"No sibling-deps inside eplinalg").
# These must be find_package'd before the baseline so its imported targets
# resolve. As of the v0.8.0 baseline all three entries are empty — the
# multifloats baseline folded its la_constants_mf/la_xisnan_mf helpers
# into the archive, and every per-precision Config find_dependency's the
# only thing it links (eplinalgStdBlas). The mechanism stays for future
# baselines that grow sibling deps.
set(EPBLAS_PARALLEL_BASELINE_DEPS_kind10       "")
set(EPBLAS_PARALLEL_BASELINE_DEPS_kind16       "")
set(EPBLAS_PARALLEL_BASELINE_DEPS_multifloats  "")

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
# overlay for the multifloats target compiles its C++ kernels against
# the `multifloats` target.
#
# Two paths into the build, picked by EPBLAS_PARALLEL_USE_SYSTEM_MULTIFLOATS:
#
#   OFF (default) — fetch + build from source (the original behavior).
#     Either point -DMULTIFLOATS_DIR at a local checkout, or fall
#     through to FetchContent at the pinned tag. The resulting archive
#     + headers + multifloatsConfig.cmake are bundled into our install
#     tree so the release tarball is self-contained.
#
#   ON — `find_package(multifloats CONFIG REQUIRED)` consumes an
#     already-installed multifloats. We do NOT bundle it; the consumer
#     of our release tarball must provide multifloats on their own
#     CMAKE_PREFIX_PATH at find_package(epblas-parallel) time. Use
#     this when CI / a host install already provides multifloats and
#     re-fetching it is wasted work.
option(EPBLAS_PARALLEL_USE_SYSTEM_MULTIFLOATS
    "Consume multifloats via find_package() instead of FetchContent"
    OFF)

# OFF path. Wrapped in a function so the BUILD_TESTING /
# MULTIFLOATS_BUILD_BENCH overrides stay function-scoped — a plain
# directory-scope `set()` would shadow the cache value of BUILD_TESTING
# for the rest of this CMakeLists.txt and silently disable the overlay
# test suite even when the user passed -DBUILD_TESTING=ON.
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
        set(MULTIFLOATS_GIT_TAG "v0.8.1"
            CACHE STRING "Git tag/branch/commit for multifloats (>= v0.8.1)")
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

    # add_subdirectory(... EXCLUDE_FROM_ALL) excludes multifloats from
    # the default build (good — keeps its tests/benches and the
    # Fortran-bindings multifloatsf out of `make all`) but also drops
    # the subdir's `cmake_install.cmake` from the parent install
    # script, which would leave consumers with a broken
    # `find_package(multifloats)` that mwblas_parallel's
    # INTERFACE_LINK_LIBRARIES depends on. Re-chain just the C/C++
    # library's install script — the multifloats library archive +
    # headers + multifloatsConfig.cmake land in our install tree, while
    # the Fortran sibling (multifloatsf) and tests/benches stay out.
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/_mf/src/cmake_install.cmake)
endfunction()

# ON path.
function(_epblas_use_system_multifloats)
    find_package(multifloats CONFIG REQUIRED)
    message(STATUS "epblas-parallel: using system multifloats "
                   "(${multifloats_DIR}) — no bundled copy in the install tree")

    # find_package's IMPORTED target is directory-scoped by default and
    # would be invisible to src/epblas-parallel/multifloats/CMakeLists.txt
    # (consumed from a sibling directory). Promote to GLOBAL so the link
    # `target_link_libraries(mwblas_parallel PUBLIC multifloats)` resolves
    # regardless of which subdir consumes it.
    set_target_properties(multifloats::multifloats PROPERTIES IMPORTED_GLOBAL TRUE)

    # The overlay kernels' link line uses the bare name `multifloats`
    # (matches the add_subdirectory case, where no namespace alias
    # exists in the build tree). Provide the alias here so the same
    # source compiles under either path; install(EXPORT) substitutes
    # the namespaced form in the emitted Targets file automatically.
    if(NOT TARGET multifloats)
        add_library(multifloats ALIAS multifloats::multifloats)
    endif()
endfunction()

# ── OpenMP runtime selection ──────────────────────────────────────────────
# Prefer Intel's libiomp5 as the OpenMP runtime when it's installed. It is
# GOMP-ABI compatible (so GCC-emitted GOMP_* calls resolve against it) but ships
# a lower-overhead thread pool with hot-team reuse (KMP_BLOCKTIME, default 200ms),
# which removes the libgomp futex-wakeup tax that inflates small/medium parallel
# regions — e.g. the kind10 L1 reductions thread cleanly under it with no need
# for OMP_WAIT_POLICY tuning.
#
# Mechanism: pre-seed FindOpenMP's cache vars so every find_package(OpenMP) below
# resolves the runtime library to iomp5. `-fopenmp` stays a COMPILE flag only
# (it's not added at link by the OpenMP:: imported targets), so the link line
# carries iomp5 and NOT libgomp — the two OpenMP runtimes are never mixed.
#
# Detected, never hardcoded as a requirement: falls back silently to the
# compiler default (libgomp for GCC) when iomp5 is absent. Force libgomp with
# -DEPBLAS_PREFER_IOMP5=OFF, or pin a runtime with -DOpenMP_C_LIB_NAMES=gomp.
option(EPBLAS_PREFER_IOMP5 "Use Intel libiomp5 as the OpenMP runtime when found" ON)
if(EPBLAS_PREFER_IOMP5 AND NOT DEFINED OpenMP_C_LIB_NAMES)
    find_library(EPBLAS_IOMP5_LIBRARY NAMES iomp5
        PATHS /opt/intel/oneapi/compiler/latest/lib
              /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin
        ENV LD_LIBRARY_PATH)
    if(EPBLAS_IOMP5_LIBRARY)
        message(STATUS "epblas: OpenMP runtime = Intel libiomp5 (${EPBLAS_IOMP5_LIBRARY})")
        set(OpenMP_C_FLAGS "-fopenmp" CACHE STRING "OpenMP C compile flags")
        set(OpenMP_C_LIB_NAMES "iomp5" CACHE STRING "OpenMP C runtime lib names")
        set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "OpenMP CXX compile flags")
        set(OpenMP_CXX_LIB_NAMES "iomp5" CACHE STRING "OpenMP CXX runtime lib names")
        set(OpenMP_iomp5_LIBRARY "${EPBLAS_IOMP5_LIBRARY}" CACHE FILEPATH "iomp5 path")
    else()
        message(STATUS "epblas: libiomp5 not found — OpenMP runtime = compiler default (libgomp)")
    endif()
endif()

if("multifloats" IN_LIST EPBLAS_PARALLEL_BUILT_TARGETS)
    if(EPBLAS_PARALLEL_USE_SYSTEM_MULTIFLOATS)
        _epblas_use_system_multifloats()
    else()
        _epblas_add_multifloats_subdir()
    endif()
endif()

add_subdirectory(src/epblas-parallel)
add_subdirectory(src/epblas-openblas)

# ── Composite INTERFACE targets (epblas-parallel package) ─────────
# epblas-parallel::{ey,qx,mw}blas wraps the overlay archive in
# WHOLE_ARCHIVE so every Fortran-callable symbol the overlay defines
# lands in the consumer's binary. Without WHOLE_ARCHIVE a consumer
# that chains another BLAS archive could resolve the same symbol
# from the wrong source; with it, the overlay always wins for the
# routines it covers. No serial baseline behind — a routine the
# overlay does not cover surfaces as an honest link error.
# Placeholder TU: a SHARED library needs at least one source of its own;
# the overlay archive itself is pulled in whole via --whole-archive below.
if(EPBLAS_BUILD_SHARED)
    set(_epblas_shared_dummy ${CMAKE_CURRENT_BINARY_DIR}/epblas_shared_dummy.c)
    file(WRITE ${_epblas_shared_dummy}
        "/* Placeholder TU — the overlay archive is linked in via\n"
        "   --whole-archive; this only satisfies CMake's requirement that a\n"
        "   SHARED library have at least one source file. */\n")
endif()

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
    # installed Targets file and downstream linkers would try `-lqxblas_parallel`
    # against the system search path. The `epblas-parallel::` form resolves to
    # the IMPORTED target the export installs alongside the composite.
    target_link_libraries(${_composite} INTERFACE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,epblas-parallel::${_overlay}>")

    message(STATUS "composite: epblas-parallel::${_composite} = WHOLE_ARCHIVE(epblas-parallel::${_overlay})")

    # Optional shared sibling: lib${_overlay}.so carrying the same
    # Fortran-callable symbols. --whole-archive forces every object of the
    # static overlay into the .so; the overlay's usage requirements
    # (OpenMP, and multifloats for the mw target) propagate through the
    # generator expression, and OpenMP::OpenMP_C is added directly for the
    # same reason the composite does.
    if(EPBLAS_BUILD_SHARED)
        set(_shared "${_prefix}blas_shared")
        add_library(${_shared} SHARED ${_epblas_shared_dummy})
        add_library(epblas-parallel::${_shared} ALIAS ${_shared})
        set_target_properties(${_shared} PROPERTIES
            OUTPUT_NAME ${_overlay}
            VERSION     ${PROJECT_VERSION}
            SOVERSION   ${PROJECT_VERSION_MAJOR})
        target_link_libraries(${_shared} PRIVATE
            "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_overlay}>")
        if(OpenMP_C_FOUND)
            target_link_libraries(${_shared} PRIVATE OpenMP::OpenMP_C)
        endif()
        message(STATUS "shared: epblas-parallel::${_shared} = lib${_overlay}.so (WHOLE_ARCHIVE ${_overlay})")
    endif()
endforeach()

# ── Install + export ──────────────────────────────────────────────
#
# Target and library file names follow the eplinalg convention
# (v0.8.0):
#   * type prefix is the real+complex pair, and target name = file
#     name — `eyblas_parallel` builds `libeyblas_parallel.a`,
#     exported as `epblas-parallel::eyblas_parallel` (cf. eplinalg's
#     `eplinalg::eyblas` / `libeyblas-gfortran-15.a`).
#   * NO compiler tag — a name carries only the tags of toolchains
#     that compile into the binary, and every archive here is pure
#     C/C++ (cf. eplinalg's `libeypblas.a`, `libmultifloats.a`). The
#     overlays implement the gfortran BLAS ABI but contain no
#     Fortran-compiled objects. The Targets exports are likewise
#     untagged (`epblas-parallelTargets.cmake`).
#
# Builds two packages out of one configure:
#   epblas-parallel — composite INTERFACEs + raw overlay archives
#   epblas-openblas — raw openblas-derived reference archives
# Excludes the `_migrated` IMPORTED archives from both — they're
# build-tree-only (test consumption), never installed.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Public version header, generated from the VERSION file via PROJECT_VERSION.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/epblas-parallel/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/epblas-parallel/version.h
    @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/epblas-parallel/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/epblas-parallel)

# User-facing documentation ships with every install (and therefore every
# release tarball): the top-level project docs plus the Markdown user guide.
# Rendered HTML (Sphinx + Doxygen, built by ./doc/build.sh) is published to
# GitHub Pages by .github/workflows/docs.yml, not an install component.
install(FILES README.md LICENSE CHANGELOG.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY doc/user/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/user
    FILES_MATCHING PATTERN "*.md")

set(_epblas_parallel_install_targets "")
set(_epblas_openblas_install_targets "")
foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
    set(_prefix "${EPBLAS_PARALLEL_PREFIX_${_target}}")
    foreach(_t ${_prefix}blas_parallel ${_prefix}blas ${_prefix}blas_shared)  # raw + composite + shared
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
        FILE epblas-parallelTargets.cmake
        NAMESPACE epblas-parallel::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-parallel
    )

    # OpenMP components needed by the consumer. C is always required;
    # CXX is required only when the multifloats target is built —
    # multifloats's C++ kernels PUBLIC-link OpenMP::OpenMP_CXX, so its
    # IMPORTED target must exist on the consumer side.
    set(EPBLAS_PARALLEL_OPENMP_COMPONENTS "C")
    set(EPBLAS_PARALLEL_EXTRA_DEPENDENCIES "")
    if("multifloats" IN_LIST EPBLAS_PARALLEL_BUILT_TARGETS)
        string(APPEND EPBLAS_PARALLEL_OPENMP_COMPONENTS " CXX")
        # mwblas_parallel PUBLIC-links multifloats::multifloats; the
        # IMPORTED target must exist on the consumer side or the
        # Targets file fails to load. multifloats is installed
        # alongside (see _epblas_add_multifloats_subdir above).
        set(EPBLAS_PARALLEL_EXTRA_DEPENDENCIES "find_dependency(multifloats)")
    endif()

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/epblas-parallelConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-parallelConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-parallel
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-parallelConfig.cmake
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
        FILE epblas-openblasTargets.cmake
        NAMESPACE epblas-openblas::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-openblas
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/epblas-openblasConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-openblasConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-openblas
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/epblas-openblasConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/epblas-openblas
    )
endif()

# Overlay test suite. Needs the fypp executable on PATH (the
# consistency / fuzz / bench drivers are .fypp-templated Fortran).
# Fortran itself is already enabled at the top of this file.
option(BUILD_TESTING "Build the overlay test suite (needs fypp + eplinalg)" ON)
if(BUILD_TESTING)
    enable_testing()

    # Pull the prebuilt eplinalg migrated baseline (eyblas/qxblas/mwblas) for each
    # built precision into the build tree and add it to CMAKE_PREFIX_PATH, so
    # the find_package() probes below resolve without a hand-staged install.
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchEplinalgBaseline.cmake)
    epblas_fetch_eplinalg_baseline()

    # Tests + benches A/B against the migrated baseline, so each
    # built target needs its eplinalg per-target package. Targets
    # without one configure cleanly but contribute no tests.
    # `find_package(<prefix>blas)` transitively imports
    # `eplinalg::blas` (std-precision, carries lsame_/xerbla_) via
    # the per-target Config's find_dependency(eplinalgStdBlas), so
    # no separate probe is needed.
    set(EPBLAS_PARALLEL_TESTABLE_TARGETS "")
    foreach(_target IN LISTS EPBLAS_PARALLEL_BUILT_TARGETS)
        set(_pkg "${EPBLAS_PARALLEL_PREFIX_${_target}}blas")
        # Pre-find any baseline helper packages so the baseline's imported
        # targets resolve (see EPBLAS_PARALLEL_BASELINE_DEPS_* above).
        foreach(_dep IN LISTS EPBLAS_PARALLEL_BASELINE_DEPS_${_target})
            find_package(${_dep} QUIET)
        endforeach()
        find_package(${_pkg} QUIET)
        if(${_pkg}_FOUND)
            message(STATUS "epblas-parallel: tests enabled for target ${_target} (${_pkg} found)")
            list(APPEND EPBLAS_PARALLEL_TESTABLE_TARGETS ${_target})
        else()
            message(STATUS "epblas-parallel: ${_pkg} not on CMAKE_PREFIX_PATH — tests for ${_target} disabled")
        endif()
    endforeach()

    add_subdirectory(test)
endif()
