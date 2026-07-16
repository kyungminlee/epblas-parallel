# Installation

## Prerequisites (production build)

- A C / C++ compiler with OpenMP (`libgomp` via gcc is fine).
- A Fortran compiler — enabled at `project()` time even when
  `BUILD_TESTING=OFF` (the eplinalg-style configure-time tag derivation
  assumes a working Fortran toolchain).
- For the `multifloats` target only: network access at configure time (or a
  local `-DMULTIFLOATS_DIR=<path>`) — the double-double runtime is pulled via
  `FetchContent`. Pass `-DEPBLAS_PARALLEL_BUILD_multifloats=OFF` to skip it.

The test/bench suite has additional prerequisites — see
[`../dev/configure.md`](../dev/configure.md).

## Build

```bash
cmake -S . -B build-prod -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build-prod -j8
```

For every target with a `src/epblas-parallel/<target>/` directory this produces
the static archive `lib{ey,qx,mw}blas_parallel.a` plus the INTERFACE composite
`epblas-parallel::{ey,qx,mw}blas`. No `eplinalg` install is consulted for a
production build.

Common configure switches (full table in
[`../dev/configure.md`](../dev/configure.md)):

| Option | Default | Effect |
|---|---|---|
| `EPBLAS_BUILD_SHARED` | `OFF` | also emit shared `.so` overlays. |
| `EPBLAS_POSITION_INDEPENDENT_CODE` | `ON` | `-fPIC` archives (`OFF` = exe-only). |
| `EPBLAS_PARALLEL_BUILD_multifloats` | `ON` | build the double-double target. |

## Install

```bash
cmake --install build-prod --prefix /opt/epblas
```

Installs both packages under the prefix:
- `epblas-parallelConfig.cmake` + targets under `lib/cmake/epblas-parallel/`
- `epblas-openblasConfig.cmake` + targets under `lib/cmake/epblas-openblas/`
- the per-target overlay archives, the composite INTERFACE targets, and the
  OpenBLAS-derived reference archives under `lib/`.

There is no version constraint on `eplinalg` — see
[usage.md](usage.md) for how the drop-in replacement relates to the baseline.
