# epblas-parallel

Hand-written extended-precision BLAS overlays for the
[eplinalg](../eplinalg/) stack.

`eplinalg` produces serial extended-precision archives by mechanical
type-rewrite of Netlib BLAS/LAPACK source — correct but unblocked and serial.
This repo layers cache-blocked, OpenMP-parallel kernels on top without touching
the eplinalg pipeline. Two libraries ship side-by-side as **separate CMake
packages**:

- **`epblas-parallel`** (the primary; same name as this repo) — production
  overlay. C/C++/OpenMP kernels for every routine across all three targets
  (`kind10`, `kind16`, `multifloats`). Ships the per-precision composite
  `epblas-parallel::{ey,qx,mw}blas` as a drop-in replacement for
  `eplinalg::{ey,qx,mw}blas`.
- **`epblas-openblas`** — experimental reference library (OpenBLAS D/Z port to
  extended precision), used purely as an A/B comparison subject.

## Quick start

```bash
# Production (no eplinalg, no tests):
cmake -S . -B build-prod -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build-prod -j8
```

Consume it downstream:

```cmake
find_package(epblas-parallel REQUIRED)
target_link_libraries(myapp PRIVATE epblas-parallel::qxblas)  # kind16 overlay
```

## Documentation

| I want to… | Start here |
|---|---|
| **Use** the library (install, link, API) | [`doc/user/`](doc/user/index.md) |
| **Install** it | [`doc/user/installation.md`](doc/user/installation.md) |
| **Configure / build / test** it | [`doc/dev/`](doc/dev/index.md) |
| **Fix a bug** | [`doc/dev/debugging.md`](doc/dev/debugging.md) |
| **Understand the internals** | [`doc/dev/architecture.md`](doc/dev/architecture.md) |
| **Run performance benchmarks** | [`benchmark/dual/README.md`](benchmark/dual/README.md) |
| **Cut a release** | [`doc/dev/release.md`](doc/dev/release.md) |

Contributors: [`CONTRIBUTING.md`](CONTRIBUTING.md). The committed perf
scoreboard is [`doc/dev/benchmark/results.md`](doc/dev/benchmark/results.md).

## Layout

```
CMakeLists.txt
CMakePresets.json
VERSION                         ← single source of truth, read by CMake
doc/
├── user/           ← how to install, link, and call the library
├── dev/            ← how to configure, build, test, debug, release
│   └── benchmark/  ← committed perf scoreboard (results.md)
├── index.md, conf.py.in, Doxyfile.in   ← Sphinx/Doxygen scaffolding
cmake/              ← Config.cmake.in templates + Fortran/baseline helpers
include/epblas-parallel/        ← version.h.in (public version header)
src/
├── epblas-parallel/<target>/   ← the overlay kernels (the product)
└── epblas-openblas/<target>/    ← OpenBLAS reference clone (A/B only)
test/               ← fypp-templated consistency + fuzz drivers
benchmark/
├── dual/           ← in-process dual-link perf harness
├── cmp5/archive/   ← frozen historical verdict reports
├── gen_dual_harnesses.py       ← dual-harness generator
└── _perf_harness/  ← perf harness Python package
example/            ← minimal downstream consumer
```
