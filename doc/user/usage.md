# Usage — consuming from CMake

```cmake
find_package(epblas-parallel REQUIRED)

add_executable(myapp main.f90)
target_link_libraries(myapp PRIVATE epblas-parallel::qxblas)  # kind16 overlay
```

`epblas-parallel::{ey,qx,mw}blas` is an INTERFACE composite that
`--whole-archive`-wraps the overlay archive, so every Fortran-callable symbol
lands in your binary. The overlay covers the full eplinalg baseline surface
(`eplinalg::{ey,qx,mw}blas` at the pinned v0.8.0), so the composite is a
**drop-in replacement** — same symbols, faster implementations.
`find_package(epblas-parallel)` does not chase any eplinalg package; production
consumers do not need eplinalg installed.

Pick the target for your precision — see [api/](api/index.md) for the naming:

| Target | Precision | Type |
|---|---|---|
| `epblas-parallel::eyblas` | kind10 | 80-bit `long double` |
| `epblas-parallel::qxblas` | kind16 | `__float128` |
| `epblas-parallel::mwblas` | multifloats | double-double |

## Falling back to the serial baseline

To use the plain serial migrated BLAS instead, depend on eplinalg directly:

```cmake
find_package(qxblas REQUIRED)
target_link_libraries(myapp PRIVATE eplinalg::qxblas)
```

## The OpenBLAS reference (A/B only)

`epblas-openblas` is an independent reference package for A/B comparison, not a
production dependency:

```cmake
find_package(epblas-openblas REQUIRED)
target_link_libraries(bench PRIVATE epblas-openblas::eyblas)  # or ::qxblas / ::mwblas
```
