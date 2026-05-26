# Public CMake API after splitting epparablas out of fortran-migrator

When the parallel/openblas overlays moved out of `fortran-migrator`
into a separate repo, we had to decide how downstream consumers ask
for an overlay-equipped BLAS. We picked **distinct namespaces with
the composite owned by the overlay project** over two alternatives:

- _Shadow_ — both packages install the same `eblas` name; one wins by
  `CMAKE_PREFIX_PATH` order. Rejected because the meaning of `eblas`
  then depends on filesystem layout, not on what the consumer typed.
  This is the closest CMake offers to apt's `Provides:`, and we
  considered it explicitly — see the "Considered" note below.
- _User composes_ — migrator ships `eblas` (serial), overlay project
  ships `eblas_parallel` (overlay), end users list both on their
  link line in the right order. Rejected because the
  overlay-before-serial link-order invariant — required for gfortran
  name-mangling shadowing to work — is exactly the footgun a
  composite exists to absorb.

We also decided that the migrator's installed `*Config.cmake` files
do **not** declare imported-target dependencies on each other.
Bundling this with the API decision because the two together define
the shape of the public surface: each migrator library is reachable
in isolation; composition (overlay + serial; lapack + blas) happens
in the consumer's link line or in epparablas's composite.

## Namespaces and the overlay project name

The migrator's installed targets live under the **`epla::`**
namespace (overriding `fortran_install_library`'s per-project
default of `${PROJECT_NAME}::`). The overlay project lives under
the **`epparablas::`** namespace.

The overlay project was originally going to be called `epblas`,
but the migrator already produces an `${LIB_PREFIX}pblas` target
for the Netlib parallel BLAS — under the kind10 prefix that's
literally `epblas`, installed as a CMake package called `epblas`.
The collision was discovered during step 4 of the split (two
`epblasConfig.cmake` files would clobber each other in one install
prefix; `find_package(epblas)` would be ambiguous regardless of
namespace). The overlay project was renamed to `epparablas` — `ep`
(extended-precision) + `para` (parallel) + `blas` — to disambiguate
without disturbing the migrator's mechanical naming convention.

## Considered

CMake offers no apt-style `Provides:` mechanism for installed
packages — only `CMAKE_PREFIX_PATH` shadowing, an `ALIAS`
(build-tree only), or a CMake ≥ 3.24 dependency provider (a global
hook in the consumer, not the provider). We chose visible name
distinction over any of these.

## Consequences

- Public `epparablas::eblas` (composite, with overlay) and
  `epla::eblas` (serial) are different names. Downstream consumers
  that want the overlay must update their `find_package` / link
  lines once.
- The migrator drops the `PARALLEL_BLAS` option, the
  `_serial` / `_migrated` suffix scheme, and the INTERFACE composite
  it owned. Its `eblas` is now a plain STATIC archive.
- The migrator's own LAPACK/ScaLAPACK/MUMPS test binaries link the
  serial BLAS only. End-to-end overlay coverage lives in
  epparablas's tests.
- Each migrator package is independently installable and versionable.
  Consumers list every archive they need on their own link line;
  symbol resolution happens at final link.
