# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with 0.x
semantics (minor = feature, patch = fix).

## [Unreleased]

## [0.13.0] - 2026-07-19

### Fixed
- The multifloats `mspmv` threaded unit-stride Lower path (`n >= 256`) passed
  an absolute row base to column-relative SIMD kernels, reading ahead of each
  column and producing wrong results; it now derives the per-column packed
  base. The salted fuzz workload pins this path on every seed.
- The fuzz log-uniform size draw now includes its upper bound, and the complex
  TRMM/TRSM checkers guard against NaN-poisoned comparisons.

### Changed
- Repo-wide cleanup pass (~600 files): stale hidden-length ABI banners, dead
  helpers and phantom env knobs, and cmp5-era benchmark remnants removed;
  reduction/rotation helpers use branchless absolute value; the multifloats
  L1 reductions share a pre-initialized partial-reduce in `mf_omp`.
- The openblas-leg L3 drivers perform all pack allocations before any
  in-place prescale and fail loudly instead of silently corrupting operands;
  the parallel m/wgemm fall back to the serial path on allocation failure.
- Benchmark drivers report bare ns/call only (dead FLOP/GF-s scaffolding and
  the JSON emitter removed); the dual harness honors an `NSDIR` override.

### Added
- Shared per-kind tuning headers (`eblas_tuning.h`, `qblas_tuning.h`,
  `mblas_tuning.h`) centralizing the openblas-leg threading thresholds.
- Salted fuzz sizing steers every 8th case into the OMP-gated regime across
  48 bodies, deterministically covering formerly unreachable threaded paths.
- `cmake/EpblasKindHelpers.cmake` dedups the six per-kind CMakeLists; CI and
  the release workflow share a composite setup action; the eplinalg baseline
  pin is single-sourced in `cmake/FetchEplinalgBaseline.cmake`.

## [0.12.0] - 2026-07-17

### Changed
- Enforced the "int is boundary-only" convention across the overlay kernels:
  the 131 internal bare-`int` violations flagged by the boundary linter are now
  `ptrdiff_t` (numeric sizes/indices/strides) or `bool` (logical flags);
  fixed-width Fortran-ABI `int` is confined to `common/` (facade + `blas_omp`).
  Covers the kind16 xtrsm L-side packed driver, perf-verified neutral against
  the `int` baseline (`par` at parity with `ob` at OMP=1 and OMP=4).

### Added
- CI and the release workflow now run the static `epblas_parallel_int_boundary_guard`
  check via `ctest -L lint` (ahead of the fuzz rotation), so a bare `int`
  slipping into an overlay kernel blocks the build.

### Fixed
- The complex-TRSM fuzz body now draws sizes up to 160 and salts cases into the
  `m >= XTRSM_PACKED_MIN_M` (128) regime, so the xtrsm L-side packed driver —
  previously never reached by fuzz (`m,n <= 96`) — is exercised on every seed.

### Removed
- Retired the cmp5 harness's historical scoreboard reports.

## [0.11.0] - 2026-07-16

### Changed
- Restructured the repository to a standard C/C++/CMake layout: `tests/` →
  `test/`, `bench/` → `benchmark/`, `scripts/` folded into `benchmark/`,
  `reports/dual_scoreboard.md` → `doc/dev/benchmark/results.md`.
- The version is now sourced from a single top-level `VERSION` file; CMake and
  the generated `epblas-parallel/version.h` read from it.
- Raised the CI and release `ctest` timeout guards so a contended GitHub
  runner finishes the fuzz rotation instead of failing spuriously.

### Added
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, and editor/tooling
  config (`.clang-format`, `.clang-tidy`, `.editorconfig`).
- Developer and user documentation trees under `doc/dev/` and `doc/user/`,
  plus Sphinx/Doxygen scaffolding.

## [0.10.0]

- Two-letter `eplinalg` v0.8.0 naming convention across targets, files, and
  baseline references (`epblas-parallel::{ey,qx,mw}blas`).

Earlier releases predate this changelog; see `git tag` (`v0.4.0`–`v0.10.0`)
and the commit history.
