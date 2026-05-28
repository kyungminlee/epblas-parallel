# Public CMake API after splitting overlays out of eplinalg

When the parallel/openblas overlays moved out of `eplinalg` (then
named `fortran-migrator`) into a separate repo, we had to decide
how downstream consumers ask for an overlay-equipped BLAS. We picked
**distinct namespaces with the composite owned by the overlay
project** over two alternatives:

- _Shadow_ — both packages install the same `eblas` name; one wins by
  `CMAKE_PREFIX_PATH` order. Rejected because the meaning of `eblas`
  then depends on filesystem layout, not on what the consumer typed.
  This is the closest CMake offers to apt's `Provides:`, and we
  considered it explicitly — see the "Considered" note below.
- _User composes_ — `eplinalg` ships `eblas` (serial), overlay project
  ships `eblas_parallel` (overlay), end users list both on their
  link line in the right order. Rejected because the
  overlay-before-serial link-order invariant — required for gfortran
  name-mangling shadowing to work — is exactly the footgun a
  composite exists to absorb.

We also decided that `eplinalg`'s installed `*Config.cmake` files
do **not** declare imported-target dependencies on each other.
Bundling this with the API decision because the two together define
the shape of the public surface: each `eplinalg` library is reachable
in isolation; composition (overlay + serial; lapack + blas) happens
in the consumer's link line or in `epblas-parallel`'s composite.

## Namespaces and the overlay project name

`eplinalg`'s installed targets live under the **`eplinalg::`**
namespace (overriding `fortran_install_library`'s per-project
default of `${PROJECT_NAME}::`). The overlay repo ships **two CMake
packages**, each with its own namespace:

- **`epblas-parallel::`** — production overlay + composite (the
  primary library; same name as the repo).
- **`epblas-openblas::`** — OpenBLAS-derived reference archives
  (used for A/B comparison; never wired into a composite).

The overlay project was originally going to be a single umbrella
package called `epblas`. The collision was discovered during step 4
of the split: `eplinalg` already produces an `${LIB_PREFIX}pblas`
target for the Netlib parallel BLAS — under the kind10 prefix
that's literally `epblas`, installed as a CMake package called
`epblas`. Two `epblasConfig.cmake` files would clobber each other
in one install prefix; `find_package(epblas)` would be ambiguous
regardless of namespace.

Initial resolution (early split commits): the umbrella was named
`epparablas` (`ep` + `para` + `blas`) to invent a non-colliding
name. The repo, the brand, and the single installed CMake package
all shared that name; the per-library names were carried as suffixes
(`epparablas::eblas_parallel`, `epparablas::eblas_epopenblas`).

Current resolution (this rename): **the repo and primary library
are both named `epblas-parallel`**, and the reference library
ships as a **separate CMake package** named `epblas-openblas`.
Same underlying principle (avoid the `epblas` collision by visible
name distinction), but with two improvements:

1. The repo name names a library, not an invented umbrella —
   easier to grep for, no opaque brand to learn.
2. The two libraries are independent CMake packages, so consumers
   that need only the reference library (e.g. A/B perf benches)
   can `find_package(epblas-openblas)` without dragging in the
   production overlay or its `find_dependency` chain back into
   `eplinalg`.

## Considered

CMake offers no apt-style `Provides:` mechanism for installed
packages — only `CMAKE_PREFIX_PATH` shadowing, an `ALIAS`
(build-tree only), or a CMake ≥ 3.24 dependency provider (a global
hook in the consumer, not the provider). We chose visible name
distinction over any of these.

## Consequences

- Public `epblas-parallel::eblas` (composite, with overlay),
  `epblas-openblas::eblas` (raw OpenBLAS-derived reference), and
  `eplinalg::eblas` (raw serial migrated) are different names.
  Downstream consumers that want the overlay must update their
  `find_package` / link lines once.
- `eplinalg` drops the historical `PARALLEL_BLAS` option, the
  `_serial` / `_migrated` suffix scheme, and the INTERFACE composite
  it owned. Its `eblas` is now a plain STATIC archive.
- `eplinalg`'s own LAPACK/ScaLAPACK/MUMPS test binaries link the
  serial BLAS only. End-to-end overlay coverage lives in the overlay
  repo's tests.
- Each `eplinalg` package is independently installable and versionable.
  Consumers list every archive they need on their own link line;
  symbol resolution happens at final link.
- The internal target name for the openblas archive keeps a
  `_openblas` suffix (`${LIB_PREFIX}blas_openblas`) for per-project
  uniqueness, but `EXPORT_NAME` strips the suffix so the consumer-
  facing name is bare `epblas-openblas::eblas`. The package
  namespace already disambiguates from `epblas-parallel::eblas`.
