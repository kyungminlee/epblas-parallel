# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with 0.x
semantics (minor = feature, patch = fix).

## [Unreleased]

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
