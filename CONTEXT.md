# epblas-parallel

Repo housing hand-written extended-precision BLAS implementations.
Sibling to `eplinalg/`, which produces the serial migrated baseline
these overlays are written against.

The repo is named after its primary library (`epblas-parallel`).
A secondary OpenBLAS-derived reference library (`epblas-openblas`)
ships from the same repo as a separate CMake package — see
[[Public CMake API]] below.

The umbrella name `epblas` was deliberately *not* used. `eplinalg`'s
kind10 PBLAS install is mechanically named `${LIB_PREFIX}pblas` =
`epblas` (under `LIB_PREFIX=e` for kind10), and the two would
collide on the install-side `lib/cmake/epblas/epblasConfig.cmake`
path. The repo takes its primary library's name instead, which
sidesteps the collision entirely.

## Language

**epblas-parallel** (the repo / the package / the library):
Production parallel overlay. Hand-written C/C++ OpenMP-parallel
kernels targeting all three [[extended-precision targets]]
(`kind10`, `kind16`, `multifloats`). Ships as
`lib{e,q,m}blas_parallel.a` and as the per-precision composite
`lib{e,q,m}blas.a` (INTERFACE — see [[composite]]). Reached as
`epblas-parallel::{e,q,m}blas_parallel` (raw archive) and
`epblas-parallel::{e,q,m}blas` (composite).

Also the name of the repo and of the CMake package that exports
these targets — `find_package(epblas-parallel)`.

**epblas-openblas**:
The experimental reference library. OpenBLAS `D`/`Z` reference
kernels ported to extended precision, currently `kind10` only.
Ships as `lib{e,q,m}blas_openblas.a` and is **never** wired into
the [[composite]] — it exists solely for A/B comparisons against
[[epblas-parallel]] and the [[migrated archive]]. Reached as
`epblas-openblas::{e,q,m}blas` (bare name; the package namespace
disambiguates, no `_openblas` suffix is needed in the consumer view).
Installed as a separate CMake package: `find_package(epblas-openblas)`.

**migrated archive**:
The serial extended-precision BLAS produced by `eplinalg`
(`lib{e,q,m}blas.a`). Correctness baseline; what the overlays
substitute for at link time. Reached as `eplinalg::{e,q,m}blas`.
_Avoid_: "the migrator's BLAS", "the Fortran BLAS".

**extended-precision targets** (a.k.a. **targets**):
The set `{kind10, kind16, multifloats}` defined by `eplinalg`'s
target configs. This repo is parametric over these but does not
own them.

## Repo identity

- Separate git repository — independent issue tracker, release
  cadence, and CI from `eplinalg`.
- CMake package versions start at `0.1.0`.
- `find_package(eblas REQUIRED)` (and the `qblas` / `mblas`
  variants) has **no version constraint** on `eplinalg`. Version
  coupling would re-introduce the lifecycle entanglement the split
  exists to remove.

## Build relationship

This repo is an independent CMake project that consumes `eplinalg`
through CMake packages (`find_package(eblas ...)` etc.) — not via
`add_subdirectory` or by being staged into eplinalg's tree.
Dependency direction is strictly one-way: this repo knows about
eplinalg's exported targets; eplinalg knows nothing about this repo.

## Public CMake API

`eplinalg` and this repo live in disjoint namespaces, and this repo
itself ships **two packages** with disjoint namespaces:

- `eplinalg` exports under `eplinalg::` — `eplinalg::eblas`,
  `eplinalg::elapack`, `eplinalg::escalapack`, `eplinalg::epblas`
  (kind10 PBLAS, distinct from this project), …
- The `epblas-parallel` package exports under `epblas-parallel::` —
  `epblas-parallel::eblas` (composite), `epblas-parallel::eblas_parallel`
  (raw overlay archive), etc.
- The `epblas-openblas` package exports under `epblas-openblas::` —
  `epblas-openblas::eblas` (raw archive), etc.

There is no "magic" `eblas` name that resolves differently
depending on what is installed; each archive is reached through
the package that ships it.

**Per-library packages, per-target via install prefix.** `eplinalg`
installs one CMake package per library; the precision target is
baked into the package name (`eblas` for kind10, `qblas` for kind16,
`mblas` for multifloats).

**`eplinalg` is per-target, this repo is multi-target.** Each
`migrator stage --target T` in eplinalg produces one target's install
prefix. This repo configures *once* and builds overlays + composites
for **every** target whose eplinalg install is discoverable on
`CMAKE_PREFIX_PATH`. Targets whose eplinalg install is absent are
silently skipped — best-effort across the target set. The shipped
`epblas-parallel` and `epblas-openblas` packages export the targets
that were actually built (`epblas-parallel::eblas` for kind10,
`epblas-parallel::qblas` for kind16, `epblas-parallel::mblas` for
multifloats; openblas is kind10-only at present).

| Package | Exports | Role |
|---|---|---|
| `eblas` / `qblas` / `mblas` (eplinalg) | `eplinalg::eblas` (STATIC, serial), … | Plain migrated archive. No overlay knowledge. |
| `elapack` / `qlapack` / `mlapack` (eplinalg) | `eplinalg::elapack` (STATIC), … | Plain migrated archive. |
| _(one such per migrated library — `escalapack`, `emumps`, `epblas` (kind10 PBLAS), etc.)_ | | |
| `epblas-parallel` | `epblas-parallel::eblas_parallel` (STATIC, the overlay), `epblas-parallel::eblas` (INTERFACE [[composite]]) per built target | Production overlay + overlay-equipped composite. |
| `epblas-openblas` | `epblas-openblas::eblas` (STATIC) per built target | OpenBLAS-derived reference archive. Never wired into the composite. |

A downstream consumer that wants the overlay-equipped BLAS does
`find_package(epblas-parallel)` and links `epblas-parallel::eblas`.
A consumer that wants the serial baseline links `eplinalg::eblas`
(after `find_package(eblas)`). A consumer that wants the openblas
reference for A/B comparison links `epblas-openblas::eblas` (after
`find_package(epblas-openblas)`). Opt-in is explicit and visible at
the call site.

### No sibling-deps inside `eplinalg` installed configs

The installed `eplinalg::*` packages do **not** declare imported-target
dependencies on each other. `find_package` for any one library
brings exactly that one library — `elapack` does not transitively
pull in `eblas`, etc. Consumers list every archive they need on
their own link line; symbol resolution happens at final link.

_Why:_ keeps each eplinalg library independently installable and
versionable; avoids coupling the install graph to the build graph.

_Implication for this repo:_ `epblas-parallel::eblas` imports
`eplinalg::eblas` *directly* — never relying on transitive linkage
through a sibling like `eplinalg::elapack`.

### Repo layout

```
epblas-parallel/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
│   ├── epblas-parallelConfig.cmake.in
│   └── epblas-openblasConfig.cmake.in
├── src/
│   ├── epblas-parallel/<target>/
│   └── epblas-openblas/<target>/
├── tests/
│   ├── epblas-parallel/
│   └── epblas-openblas/
├── doc/
├── scripts/
└── reports/
```

The `src/` layer is preserved (rather than flattening overlays to
the repo root) because the two libraries are *siblings under the
repo*, not the repo itself, and the layer leaves room for future
internal source dirs that aren't overlay kernels.

### Discovery is baked in

The top-level `CMakeLists.txt` hardcodes the full `{kind10→eblas,
kind16→qblas, multifloats→mblas}` table and enumerates targets by
doing `find_package(eblas QUIET)`, `find_package(qblas QUIET)`,
`find_package(mblas QUIET)`. `eplinalg` exports **no extra metadata**
in its installed Configs to support this — the contract surface
between the two projects is exactly the package names and the
imported-target archives, nothing more.

_Implication:_ adding a new precision target (e.g. `kind20`) is a
coordinated change touching both repos.

**composite**:
The `epblas-parallel::eblas` INTERFACE target (and `qblas`, `mblas`).
Links `epblas-parallel::eblas_parallel` *before* `eplinalg::eblas`
so overlay symbols shadow serial symbols at final link (gfortran
name-mangling parity). The link-order invariant is owned by this
INTERFACE; consumers never reproduce it.

## Out of scope for `eplinalg` after the split

The split is governed by a single rule:

> **Anything that doesn't follow the eplinalg pipeline moves to
> this repo.**

A thing "follows the eplinalg pipeline" iff it is mechanically
derived from upstream Fortran source by the migrator CLI, or it
directly serves that derivation (recipes, targets, vendored
upstreams, the staging build system, tests of migrated archives,
pipeline docs). Anything else — hand-written kernels, ported
kernels, overlay tests, overlay perf tooling, overlay docs —
moves.

Concrete consequences:

- `eplinalg`'s `eblas` target collapses to the plain serial archive
  (today's `${LIB_PREFIX}blas_serial`). The `_serial` / `_migrated`
  suffix gymnastics and the historical `PARALLEL_BLAS` option leave
  `eplinalg` entirely.
- `eplinalg`'s installed packages namespace their imported targets
  under `eplinalg::` (overriding the per-project default).
- `eplinalg`'s own LAPACK/ScaLAPACK/MUMPS test binaries link the
  serial BLAS only. End-to-end overlay coverage lives in this repo's
  tests.
- The stage-time copy in `src/migrator/__main__.py` that used to drop
  `parallel_blas/` and `epopenblas/` into the staging tree is
  deleted; `eplinalg` no longer references either overlay.
