# epparablas

Umbrella project housing hand-written extended-precision BLAS
implementations. Sibling to `fortran-migrator/`, which produces
the serial migrated baseline these overlays are written against.

The name is `ep` (extended-precision) + `para` (parallel) + `blas`.
It avoids the collision with the migrator's own kind10 PBLAS
target, which is mechanically named `${LIB_PREFIX}pblas` = `epblas`
under the migrator's prefix convention. The two would otherwise
both install as a CMake package called `epblas`.

## Language

**epparablas**:
The umbrella project. Not a single library — it ships multiple
ep-BLAS implementations side by side. The shipped archives keep
their own names (see [[parallel-blas]], [[epopenblas]]);
"epparablas" names the *repo*, the *brand*, and the installed
CMake package — not any one archive.
_Avoid_: using "epparablas" as the name of any single archive or
CMake target.

**parallel-blas**:
The production overlay. Hand-written C/C++ OpenMP-parallel
kernels targeting all three [[extended-precision targets]]
(`kind10`, `kind16`, `multifloats`). Ships as
`lib{e,q,m}blas_parallel.a`. Reached as
`epparablas::{e,q,m}blas_parallel`.

**epopenblas**:
The experimental overlay. OpenBLAS `D`/`Z` reference kernels
ported to extended precision, currently `kind10` only. Ships as
`lib{e,q,m}blas_epopenblas.a` and is **never** wired into the
[[composite]] — it exists solely for A/B comparisons against
[[parallel-blas]] and the [[migrated archive]].

**migrated archive**:
The serial extended-precision BLAS produced by
`fortran-migrator` (`lib{e,q,m}blas.a`). Correctness baseline;
what the overlays substitute for at link time. Reached as
`epla::{e,q,m}blas`.
_Avoid_: "the migrator's BLAS", "the Fortran BLAS".

**extended-precision targets** (a.k.a. **targets**):
The set `{kind10, kind16, multifloats}` defined by
`fortran-migrator`'s target configs. epparablas is parametric
over these but does not own them.

## Repo identity

- Separate git repository — independent issue tracker, release
  cadence, and CI from `fortran-migrator`.
- CMake package version starts at `0.1.0`.
- `find_package(eblas REQUIRED)` (and the `qblas` / `mblas`
  variants) has **no version constraint** on the migrator. Version
  coupling would re-introduce the lifecycle entanglement the split
  exists to remove.

## Build relationship

epparablas is an independent CMake project that consumes
`fortran-migrator` through CMake packages
(`find_package(eblas ...)` etc.) — not via `add_subdirectory` or
by being staged into the migrator's tree. Dependency direction is
strictly one-way: epparablas knows about the migrator's exported
targets; the migrator knows nothing about epparablas.

## Public CMake API

Migrator and epparablas live in disjoint namespaces:

- Migrator exports under `epla::` — `epla::eblas`,
  `epla::elapack`, `epla::escalapack`, `epla::epblas` (kind10
  PBLAS, distinct from this project), …
- epparablas exports under `epparablas::` — `epparablas::eblas`
  (composite), `epparablas::eblas_parallel`, etc.

There is no "magic" `eblas` name that resolves differently
depending on what is installed; each archive is reached through
the package that ships it.

**Per-library packages, per-target via install prefix.** The
migrator installs one CMake package per library; the precision
target is baked into the package name (`eblas` for kind10, `qblas`
for kind16, `mblas` for multifloats).

**Migrator is per-target, epparablas is multi-target.** Each
`migrator stage --target T` produces one target's install prefix.
epparablas configures *once* and builds overlays + composite for
**every** target whose migrator install is discoverable on
`CMAKE_PREFIX_PATH`. Targets whose migrator install is absent are
silently skipped — epparablas is best-effort across the target
set. The shipped `epparablas` package exports composites for the
targets that were actually built (`epparablas::eblas` for kind10,
`epparablas::qblas` for kind16, `epparablas::mblas` for
multifloats).

| Package | Exports | Role |
|---|---|---|
| `eblas` / `qblas` / `mblas` (migrator) | `epla::eblas` (STATIC, serial), … | Plain migrated archive. No overlay knowledge. |
| `elapack` / `qlapack` / `mlapack` (migrator) | `epla::elapack` (STATIC), … | Plain migrated archive. |
| _(one such per migrated library — `escalapack`, `emumps`, `epblas` (kind10 PBLAS), etc.)_ | | |
| `epparablas` (one per target prefix) | `epparablas::eblas_parallel` (STATIC, the overlay), `epparablas::eblas_epopenblas` (STATIC, kind10 only), `epparablas::eblas` (INTERFACE [[composite]]) | Overlays + composite. |

A downstream consumer that wants the overlay-equipped BLAS does
`find_package(epparablas)` and links `epparablas::eblas`. A
consumer that wants the serial baseline links `epla::eblas`
(after `find_package(eblas)`). Opt-in is explicit and visible at
the call site.

### No sibling-deps inside `epla` installed configs

The installed `epla::*` packages do **not** declare imported-target
dependencies on each other. `find_package` for any one library
brings exactly that one library — `elapack` does not transitively
pull in `eblas`, etc. Consumers list every archive they need on
their own link line; symbol resolution happens at final link.

_Why:_ keeps each migrator library independently installable and
versionable; avoids coupling the install graph to the build
graph.

_Implication for epparablas:_ `epparablas::eblas` imports
`epla::eblas` *directly* — never relying on transitive linkage
through a sibling like `epla::elapack`.

### Repo layout

Mirrors the migrator's shape so files move with their references
intact:

```
epparablas/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
├── src/
│   ├── parallel_blas/<target>/
│   └── epopenblas/<target>/
├── tests/
│   ├── blas_parallel/
│   └── blas_epopenblas/
├── doc/
├── scripts/
└── reports/
```

The `src/` layer is preserved (rather than flattening overlays to
the repo root) because the two overlays are *siblings under the
umbrella*, not the umbrella itself, and the layer leaves room for
future epparablas-internal source dirs that aren't overlay
kernels.

### Discovery is baked in

epparablas hardcodes the full `{kind10→eblas, kind16→qblas,
multifloats→mblas}` table and enumerates targets by doing
`find_package(eblas QUIET)`, `find_package(qblas QUIET)`,
`find_package(mblas QUIET)`. The migrator exports **no extra
metadata** in its installed Configs to support this — the contract
surface between the two projects is exactly the package names and
the imported-target archives, nothing more.

_Implication:_ adding a new precision target (e.g. `kind20`) is a
coordinated change touching both repos.

**composite**:
The `epparablas::eblas` INTERFACE target (and `qblas`, `mblas`).
Links `epparablas::eblas_parallel` *before* `epla::eblas` so
overlay symbols shadow serial symbols at final link (gfortran
name-mangling parity). The link-order invariant is owned by this
INTERFACE; consumers never reproduce it.

## Out of scope for the migrator after the split

The split is governed by a single rule:

> **Anything that doesn't follow the migrator pipeline moves to
> epparablas.**

A thing "follows the migrator pipeline" iff it is mechanically
derived from upstream Fortran source by the migrator, or it
directly serves that derivation (recipes, targets, vendored
upstreams, the staging build system, tests of migrated
archives, pipeline docs). Anything else — hand-written kernels,
ported kernels, overlay tests, overlay perf tooling, overlay
docs — moves.

Concrete consequences:

- The migrator's `eblas` target collapses to the plain serial
  archive (today's `${LIB_PREFIX}blas_serial`). The
  `_serial` / `_migrated` suffix gymnastics and the
  `PARALLEL_BLAS` option leave the migrator entirely.
- The migrator's installed packages namespace their imported
  targets under `epla::` (overriding the per-project default).
- The migrator's own LAPACK/ScaLAPACK/MUMPS test binaries link
  the serial BLAS only. End-to-end overlay coverage lives in
  epparablas's tests.
- The stage-time copy in `src/migrator/__main__.py` that drops
  `parallel_blas/` and `epopenblas/` into the staging tree is
  deleted; the migrator no longer references either overlay.
