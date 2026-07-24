# Using epblas-parallel

`epblas-parallel` provides hand-written, cache-blocked, OpenMP-parallel
extended-precision BLAS overlays — a drop-in replacement for the serial
[eplinalg](https://github.com/kyungminlee/eplinalg) baseline across three
precisions: **kind10**
(80-bit `long double`), **kind16** (`__float128`), and **multifloats**
(double-double).

- **[installation.md](installation.md)** — prerequisites, build, install.
- **[usage.md](usage.md)** — link it from your CMake project.
- **[api/](api/index.md)** — library/target naming and the routine surface.

Building or extending the library itself? See the developer docs in
[`../dev/`](../dev/index.md).

```{toctree}
:hidden:

installation
usage
api/index
```
