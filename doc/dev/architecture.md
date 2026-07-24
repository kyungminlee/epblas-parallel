# Architecture

## Two packages, side by side

`eplinalg` produces serial extended-precision BLAS archives by mechanical
type-rewrite of Netlib source — correct but unblocked and serial. This repo
layers cache-blocked, OpenMP-parallel kernels on top without touching the
eplinalg pipeline, and ships two independent CMake packages:

- **`epblas-parallel`** — the production overlay. Hand-written C/C++/OpenMP
  kernels for every routine across all three targets. Exports the composite
  `epblas-parallel::{ey,qx,mw}blas`.
- **`epblas-openblas`** — an OpenBLAS D/Z port to extended precision, used only
  as an A/B reference subject. Not wired into the public composite.

Neither package version-constrains `eplinalg`; the split exists precisely to
remove that lifecycle entanglement.

## Three precision targets

| Target | Dir | Scalar | Real / complex prefix | Source ext |
|---|---|---|---|---|
| kind10 | `src/epblas-parallel/kind10/` | 80-bit `long double` | `e` / `y` | `.c` |
| kind16 | `src/epblas-parallel/kind16/` | `__float128` | `q` / `x` | `.c` |
| multifloats | `src/epblas-parallel/multifloats/` | double-double | `m` / `w` | `.cpp` |

kind16 is a faithful hand-port of kind10 (a closed type-rewrite set);
multifloats is C++ over the double-double limb type and is the only family with
SIMD (AVX2 SoA on the limbs). A target with no source directory is simply
skipped at configure time.

## Per-routine module layout

A routine is either a single `<name>.c` (simple L1) or a triad:

| File | Role |
|---|---|
| `<name>_serial.c` | the OMP=1 path — the serial baseline, kept bit-exact with netlib. |
| `<name>_parallel.c` | the threaded path; dispatches to the serial path below a per-routine size threshold, and must match it bit-for-bit above it. |
| `<name>_kernel.h` | the microkernel both paths `#include`. |

The threshold model matters: parallelism only pays above a routine- and
shape-dependent size, so each `_parallel.c` guards the threaded region and falls
through to the serial kernel for small problems. Getting a threshold wrong shows
up as an OMP=4 regression in the perf scoreboard, not a correctness failure.

## The overlay linking model

The top-level `CMakeLists.txt` builds each `_parallel` archive, then wraps it in
an INTERFACE composite `epblas-parallel::{ey,qx,mw}blas`. The composite links
the `_parallel` archive **`--whole-archive`** so every Fortran-callable symbol
the overlay defines is pulled into the consumer's binary and *wins* over the
serial baseline's same-named symbol. The overlay covers the full baseline
surface, so the composite is a drop-in replacement for `eplinalg::{ey,qx,mw}blas`
— consumers relink, nothing else changes. (This is the model the source
comments refer to; it supersedes the historical `doc/design.md`.)

The perf harness relies on the same "same symbols, different archives" property
in reverse: it namespaces par/ob/mig archives with `objcopy` so all three
coexist in one binary — see
[`benchmark/dual/BENCH_PROTOCOL.md`](https://github.com/kyungminlee/epblas-parallel/blob/main/benchmark/dual/BENCH_PROTOCOL.md).

## OpenMP runtime

With `EPBLAS_PREFER_IOMP5=ON` (default) the build resolves the OpenMP runtime to
Intel `libiomp5` when present — GOMP-ABI compatible, and its hot-team reuse cuts
the thread-wakeup tax that dominates small threaded calls. `-fopenmp` stays a
compile-only flag so the two runtimes are never mixed at link.

## Repository map

```
include/epblas-parallel/         public version header (version.h.in)
src/epblas-parallel/<target>/    overlay kernels (the product)
src/epblas-openblas/<target>/    OpenBLAS reference clone (A/B only)
test/epblas-parallel/            fypp-templated consistency + fuzz drivers
test/epblas-openblas/            same, reusing the parallel bodies
benchmark/drivers/target_<target>/   C/C++ perf drivers
benchmark/dual/                      in-process dual-link sweep harness
benchmark/gen_dual_harnesses.py      dual-harness generator
benchmark/_perf_harness/             perf harness Python package
doc/dev/benchmark/results.md         committed perf scoreboard
example/                         minimal downstream consumer
cmake/                           Config.cmake.in templates + Fortran/baseline helpers
```
