# epblas-parallel

Hand-written extended-precision BLAS overlays for the
[eplinalg](../eplinalg/) stack.

`eplinalg` produces serial extended-precision archives by mechanical
type-rewrite of Netlib BLAS/LAPACK source — correct but unblocked and
serial. This repo layers cache-blocked, OpenMP-parallel kernels on top
without touching the eplinalg pipeline. Two libraries ship side-by-side
as **separate CMake packages**:

- **`epblas-parallel`** (the primary; same name as this repo) — production
  overlay. C/C++/OpenMP kernels for every routine, all three targets
  (`kind10`, `kind16`, `multifloats`). Ships the per-precision composite
  `epblas-parallel::{ey,qx,mw}blas` as a drop-in replacement for
  `eplinalg::{ey,qx,mw}blas`.
- **`epblas-openblas`** — experimental reference library. OpenBLAS D/Z
  port to extended precision, all three targets (`kind10`, `kind16`,
  `multifloats`). Used purely as an A/B comparison subject against
  `epblas-parallel` and the migrated baseline.

## Prerequisites

### Production build (`-DBUILD_TESTING=OFF`)

- A C / C++ compiler with OpenMP (`libgomp` via gcc is fine).
- A Fortran compiler — enabled at project() time even when
  `BUILD_TESTING=OFF` (the configure-time tag-derivation used by
  eplinalg-style packages assumes a working Fortran toolchain).
- For the `multifloats` target only: an internet connection at
  configure time (or a local `MULTIFLOATS_DIR`). This repo fetches the
  `multifloats` C/Fortran library via `FetchContent`. Pass
  `-DEPBLAS_PARALLEL_BUILD_multifloats=OFF` to skip.

### Additional, for the test / bench suite (`-DBUILD_TESTING=ON`, default)

- `fypp` on `PATH` — the consistency / fuzz / bench drivers are
  `.fypp`-templated Fortran.
- `eplinalg` installed for each target you want exercised. Tests A/B
  the overlay against eplinalg's migrated baseline. Each install must
  be reachable via `CMAKE_PREFIX_PATH` at configure time. A target
  with no eplinalg install configures cleanly but contributes no tests.

## Quick start (production)

```bash
# In epblas-parallel/:
cmake -S . -B build-prod \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF
cmake --build build-prod -j8
```

This produces, for every target whose `src/epblas-parallel/<target>/`
directory exists, the static archive `lib{ey,qx,mw}blas_parallel.a` plus
the INTERFACE composite `epblas-parallel::{ey,qx,mw}blas`. No `eplinalg`
install is consulted.

## Running the tests / bench

Install `eplinalg` for one or more targets:

```bash
# In the eplinalg/ repo:
uv run python -m migrator stage /tmp/stage-q --target kind16
cmake -S /tmp/stage-q -B /tmp/stage-q/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/stage-q/build -j8
cmake --install /tmp/stage-q/build --prefix /opt/eplinalg-q
```

Then point `CMAKE_PREFIX_PATH` at every install prefix you want exercised
and pick the workflow preset matching the cadence you want.

### Workflow presets

`CMakePresets.json` ships four `cmake --workflow` chains
(configure → build → ctest with a label filter):

| Preset     | Tests run                                            | Wall-clock                |
|------------|------------------------------------------------------|---------------------------|
| `fuzz`     | All fuzz / consistency drivers (the correctness gate) | ~15 s                     |
| `perf`     | C `perf_*` harnesses (jobs=1, 1800 s/test cap)       | tens of minutes per slice |
| `sweep`    | `dual_sweep_{e,q,m}` in-process dual-link drivers    | hours                     |
| `e2e`      | Everything above                                     | overnight                 |

Everyday loop:

```bash
CMAKE_PREFIX_PATH=/opt/eplinalg-q cmake --workflow --preset fuzz
```

The fuzz workflow is the correctness gate — run it after every code
change. Re-run a slice without rebuilding via the matching `ctest`
preset:

```bash
ctest --preset fuzz       # re-runs label=fuzz only
ctest --preset perf -N    # dry-run: list what would be run
```

`perf` and `sweep` are gated as their own presets so the everyday
`fuzz` run stays cheap. The `sweep` preset runs the in-process
namespaced dual-link harness `bench/dual/run_dual.sh` once per precision
family (`e`/`q`/`m`): par, ob, and mig are linked into one binary and
timed interleaved per rep at OMP=1 and OMP=4. Raw data lands in the
gitignored `workspace/files/gap5/nsbench/`; the scoreboard is committed to
`reports/dual_scoreboard.md`.

To run perf directly (re-time a routine, sweep a family, refresh the
scoreboard) start with the runbook **`bench/dual/README.md`** — it covers
prerequisites, the env knobs, how to read the board, and the machine
hygiene that keeps the numbers trustworthy. `bench/dual/BENCH_PROTOCOL.md`
is the rationale behind the method.

Build-only (skip the workflow chaining):

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/eplinalg-q  # or /opt/eplinalg-e;/opt/eplinalg-q;/opt/eplinalg-m
cmake --build build -j8
```

The configure log reports which targets are testable:

```
-- epblas-parallel: building target kind16
-- epblas-parallel: tests enabled for target kind16 (qxblas found)
-- epblas-parallel: eyblas not on CMAKE_PREFIX_PATH — tests for kind10 disabled
```

Tests/bench A/B against eplinalg's migrated baseline; the production
build does not link it.

## Consuming from a downstream CMake project

```cmake
find_package(epblas-parallel REQUIRED)

add_executable(myapp main.f90)
target_link_libraries(myapp PRIVATE epblas-parallel::qxblas)  # kind16, with overlay
```

`epblas-parallel::{ey,qx,mw}blas` is an INTERFACE composite that
WHOLE_ARCHIVE-wraps the overlay archive. The overlay covers the full
eplinalg baseline surface (`eplinalg::{ey,qx,mw}blas` at the pinned
v0.8.0), so the composite is a drop-in replacement. `find_package(epblas-parallel)` does not chase any
eplinalg package — production consumers do not need eplinalg
installed.

To use the plain serial migrated BLAS instead, depend on eplinalg's
package directly: `find_package(qxblas); target_link_libraries(myapp PRIVATE eplinalg::qxblas)`.

To use the OpenBLAS-derived reference archive for A/B comparison:

```cmake
find_package(epblas-openblas REQUIRED)
target_link_libraries(bench PRIVATE epblas-openblas::eyblas)  # or ::qxblas / ::mwblas
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

No version constraint on `eplinalg` — version coupling would re-introduce
the lifecycle entanglement the split exists to remove.

## Layout

```
CMakeLists.txt
CMakePresets.json
cmake/
├── FortranCompiler.cmake        ← copy of eplinalg's helper (sync by hand)
├── FetchEplinalgBaseline.cmake  ← fetches the migrated baseline release binaries
├── epblas-parallelConfig.cmake.in
└── epblas-openblasConfig.cmake.in
src/
├── epblas-parallel/<target>/
└── epblas-openblas/<target>/
tests/
├── epblas-parallel/        ← consistency + fuzz
└── epblas-openblas/        ← consistency + fuzz, reuses bodies from epblas-parallel
bench/
├── drivers/target_<target>/  ← C/C++ perf drivers (shared by both suites)
├── dual/                   ← in-process dual-link sweep harness + scoreboard tools
└── cmp5/archive/           ← frozen historical cmp5 verdict reports
scripts/                  ← dual-harness generator + report utilities
```
