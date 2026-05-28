# epblas-parallel

Hand-written extended-precision BLAS overlays for the
[eplinalg](../fortran-migrator/) stack.

`eplinalg` produces serial extended-precision archives by mechanical
type-rewrite of Netlib BLAS/LAPACK source — correct but unblocked and
serial. This repo layers cache-blocked, OpenMP-parallel kernels on top
without touching the eplinalg pipeline. Two libraries ship side-by-side
as **separate CMake packages**:

- **`epblas-parallel`** (the primary; same name as this repo) — production
  overlay. C/C++/OpenMP kernels for every routine, all three targets
  (`kind10`, `kind16`, `multifloats`). Ships the per-precision composite
  `epblas-parallel::{e,q,m}blas` (overlay + serial baseline). See
  `doc/design.md`.
- **`epblas-openblas`** — experimental reference library. OpenBLAS D/Z
  port to extended precision, kind10 only. Used purely as an A/B
  comparison subject against `epblas-parallel` and the migrated baseline.

For the umbrella terminology, the migrator/overlay split, and the
public-API decisions, see `CONTEXT.md` and `docs/adr/0001-public-cmake-api-after-split.md`.

## Prerequisites

- A C / C++ compiler with OpenMP (`libgomp` via gcc is fine).
- A Fortran compiler (`gfortran`) **if** you build the test suite
  (`BUILD_TESTING=ON`, default). The drivers are `.fypp`-templated
  Fortran.
- `fypp` on `PATH` for the same reason.
- `eplinalg` installed for at least one target. Each install must be
  reachable via `CMAKE_PREFIX_PATH` at configure time.
- For the `multifloats` target only: an internet connection at
  configure time (or a local `MULTIFLOATS_DIR`). This repo fetches the
  `multifloats` C/Fortran library via `FetchContent`.

## Quick start

Install `eplinalg` for one or more targets:

```bash
# In the eplinalg/ repo:
uv run python -m migrator stage /tmp/stage-q --target kind16
cmake -S /tmp/stage-q -B /tmp/stage-q/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/stage-q/build -j8
cmake --install /tmp/stage-q/build --prefix /opt/eplinalg-q
```

Configure this repo pointing at every install prefix you want
overlays for:

```bash
# In epblas-parallel/:
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/eplinalg-q  # or /opt/eplinalg-e;/opt/eplinalg-q;/opt/eplinalg-m
cmake --build build -j8
ctest --test-dir build
```

The configure log reports which targets it found:

```
-- epblas-parallel: found qblas (target kind16) — overlay will build
-- epblas-parallel: eblas not found — skipping target kind10
-- epblas-parallel: mblas not found — skipping target multifloats
```

Targets without a corresponding eplinalg install are silently skipped
— best-effort across the target set.

## Consuming from a downstream CMake project

```cmake
find_package(epblas-parallel REQUIRED)

add_executable(myapp main.f90)
target_link_libraries(myapp PRIVATE epblas-parallel::qblas)  # kind16, with overlay
```

`epblas-parallel::<prefix>blas` is an INTERFACE composite that links
the overlay archive *before* eplinalg's serial archive
(`eplinalg::<prefix>blas`), so overlay symbols shadow serial symbols
at final link. `find_package(epblas-parallel)` brings in
`find_dependency` on each eplinalg package this build linked against
— you don't need to do those yourself.

To use the plain serial migrated BLAS instead, depend on eplinalg's
package directly: `find_package(qblas); target_link_libraries(myapp PRIVATE eplinalg::qblas)`.

To use the OpenBLAS-derived reference archive for A/B comparison:

```cmake
find_package(epblas-openblas REQUIRED)
target_link_libraries(bench PRIVATE epblas-openblas::eblas)  # kind10 only
```

The reference package is independent — pull it in only if you need it.

## Install

```bash
cmake --install build --prefix /opt/epblas
```

Installs both packages:
- `epblas-parallelConfig.cmake` + targets file under `lib/cmake/epblas-parallel/`
- `epblas-openblasConfig.cmake` + targets file under `lib/cmake/epblas-openblas/`
- The per-target overlay archives, the composite INTERFACE targets,
  and the openblas reference archives under `lib/`.

No version constraint on `eplinalg` — see `CONTEXT.md` §"Repo identity".

## Layout

```
CMakeLists.txt
CMakePresets.json
CONTEXT.md                ← glossary + design decisions
cmake/
├── FortranCompiler.cmake     ← copy of eplinalg's helper (sync by hand)
├── epblas-parallelConfig.cmake.in
└── epblas-openblasConfig.cmake.in
docs/adr/                 ← architectural decision records
doc/                      ← design and optimization-findings docs
src/
├── epblas-parallel/<target>/
└── epblas-openblas/<target>/  ← kind10 only
tests/
├── epblas-parallel/        ← consistency + bench + perf
└── epblas-openblas/        ← consistency + bench + perf, reuses bodies from epblas-parallel
scripts/                  ← perf-sweep + report-generation utilities
reports/cmp5/             ← perf comparison artifacts
```
