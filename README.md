# epparablas

Hand-written extended-precision BLAS overlays for the
[fortran-migrator](../fortran-migrator/) stack.

The migrator produces serial extended-precision archives by mechanical
type-rewrite of Netlib BLAS/LAPACK source — correct but unblocked and
serial. epparablas layers cache-blocked, OpenMP-parallel kernels on top
without touching the migrator pipeline. Two overlay implementations
ship side-by-side:

- **parallel-blas** — the production overlay. C/C++/OpenMP kernels for
  every routine, all three targets (`kind10`, `kind16`, `multifloats`).
  See `doc/design.md`.
- **epopenblas** — an experimental OpenBLAS-D/Z-port to extended
  precision. Kind10 only. Used purely as an A/B comparison subject
  against `parallel-blas` and the migrated baseline.

For the umbrella terminology, the migrator/overlay split, and the
public-API decisions, see `CONTEXT.md` and `docs/adr/0001-public-cmake-api-after-split.md`.

## Prerequisites

- A C / C++ compiler with OpenMP (`libgomp` via gcc is fine).
- A Fortran compiler (`gfortran`) **if** you build the test suite
  (`BUILD_TESTING=ON`, default). The drivers are `.fypp`-templated
  Fortran.
- `fypp` on `PATH` for the same reason.
- `fortran-migrator` installed for at least one target. Each install
  must be reachable via `CMAKE_PREFIX_PATH` at configure time.
- For the `multifloats` target only: an internet connection at
  configure time (or a local `MULTIFLOATS_DIR`). epparablas fetches
  the `multifloats` C/Fortran library via `FetchContent`.

## Quick start

Install the migrator for one or more targets:

```bash
# In the fortran-migrator/ repo:
uv run python -m migrator stage /tmp/stage-q --target kind16
cmake -S /tmp/stage-q -B /tmp/stage-q/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/stage-q/build -j8
cmake --install /tmp/stage-q/build --prefix /opt/epla-q
```

Configure epparablas pointing at every install prefix you want
overlays for:

```bash
# In epparablas/:
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/epla-q  # or /opt/epla-e;/opt/epla-q;/opt/epla-m
cmake --build build -j8
ctest --test-dir build
```

The configure log reports which targets it found:

```
-- epparablas: found qblas (target kind16) — overlay will build
-- epparablas: eblas not found — skipping target kind10
-- epparablas: mblas not found — skipping target multifloats
```

Targets without a corresponding migrator install are silently skipped
— epparablas is best-effort across the target set.

## Consuming from a downstream CMake project

```cmake
find_package(epparablas REQUIRED)

add_executable(myapp main.f90)
target_link_libraries(myapp PRIVATE epparablas::qblas)  # kind16, with overlay
```

`epparablas::<prefix>blas` is an INTERFACE composite that links the
overlay archive *before* the migrator's serial archive
(`epla::<prefix>blas`), so overlay symbols shadow serial symbols at
final link. `find_package(epparablas)` brings in `find_dependency` on
each migrator package this build linked against — you don't need to
do those yourself.

To use the plain serial migrated BLAS instead, depend on the migrator's
package directly: `find_package(qblas); target_link_libraries(myapp PRIVATE epla::qblas)`.

## Install

```bash
cmake --install build --prefix /opt/epparablas
```

Installs `epparablasConfig.cmake`, `epparablasConfigVersion.cmake`, the
per-target overlay archives, and the composite INTERFACE targets. No
version constraint on the migrator — see `CONTEXT.md` §"Repo identity".

## Layout

```
CMakeLists.txt
CMakePresets.json
CONTEXT.md                ← glossary + design decisions
cmake/
├── FortranCompiler.cmake     ← copy of the migrator's helper (sync by hand)
└── epparablasConfig.cmake.in
docs/adr/                 ← architectural decision records
doc/                      ← design and optimization-findings docs
src/
├── parallel_blas/<target>/
└── epopenblas/<target>/  ← kind10 only
tests/
├── blas_parallel/        ← consistency + bench + perf
└── blas_epopenblas/      ← consistency + bench + perf, reuses bodies from blas_parallel
scripts/                  ← perf-sweep + report-generation utilities
reports/cmp5/             ← perf comparison artifacts
```
