# Build

## Commands

Production (no eplinalg, no tests):

```bash
cmake -S . -B build-prod -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build-prod -j8
```

With tests/bench (the default; needs `fypp` + an eplinalg reference on
`CMAKE_PREFIX_PATH` — see [configure.md](configure.md)):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/eplinalg-q
cmake --build build -j8
```

## Artifacts

For each target with a `src/epblas-parallel/<target>/` directory:

- `lib{ey,qx,mw}blas_parallel.a` — the tuned static overlay archive.
- `epblas-parallel::{ey,qx,mw}blas` — the INTERFACE composite that
  `--whole-archive`-wraps the overlay (this is what consumers link; see
  [architecture.md](architecture.md) for the linking model).
- With `-DEPBLAS_BUILD_SHARED=ON`, matching `.so` overlays as well.

The `epblas-openblas` reference archives build alongside but are not wired into
the public composite.

## Generated test/bench code

There is **no code generator for the library** — kernels are hand-written C/C++
under `src/`. Two kinds of generated code exist, both test/bench-only, and both
run automatically as part of the build:

- **fypp templates** (`tests/**/*.fypp`) — the consistency/fuzz drivers are
  Fortran expanded from fypp at build time. `fypp` must be on `PATH`; edit the
  `.fypp` template, never the generated `.f90`. Building the drivers also emits
  Fortran `.mod` files into the build tree — these are build artifacts, never
  committed.
- **Python bench-driver generator** (`scripts/gen_dual_harnesses.py`) — emits
  the per-routine dual-link perf drivers. It is invoked by the perf scripts, not
  the CMake build; to run or modify it see [test.md](test.md) and the perf
  runbook [`../../bench/dual/README.md`](../../bench/dual/README.md).

## Install

```bash
cmake --install build --prefix /opt/epblas
```

Installs both CMake packages, their targets files, the overlay archives, the
composite INTERFACE targets, and the OpenBLAS reference archives. Consumer-side
details are in [`../user/installation.md`](../user/installation.md).
