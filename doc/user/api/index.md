# API reference

`epblas-parallel` does not define a new API — it re-implements the existing
[eplinalg](../../../README.md) BLAS surface with the identical Fortran ABI, so
every entry point is called exactly as the serial baseline (and as Netlib BLAS,
retyped to the extended precision). Linking the overlay changes performance,
not call sites.

## Naming

Each precision ships as a two-letter library with its own routine prefixes:

| Precision | Library / target | Real prefix | Complex prefix | Scalar type |
|---|---|---|---|---|
| kind10 | `eyblas` | `e` | `y` | 80-bit `long double` |
| kind16 | `qxblas` | `q` | `x` | `__float128` |
| multifloats | `mwblas` | `m` | `w` | double-double |

So `egemm_` is the kind10 real GEMM, `xtrsm_` the kind16 complex TRSM, `mnrm2_`
the multifloats real 2-norm — one entry per `(precision, real/complex,
BLAS routine)`, matching the eplinalg baseline one-for-one.

## Surface

The overlay implements the full baseline surface: the BLAS **L1** (dot, axpy,
scal, nrm2, asum, swap, rot, i*amax, …), **L2** (gemv, symv/hemv, trmv/trsv,
ger, band/packed variants, rank updates), and **L3** (gemm, symm/hemm, syrk/
herk, syr2k/her2k, trmm, trsm) levels. The canonical list is whatever
`eplinalg::{ey,qx,mw}blas` exports at the pinned baseline version; the overlay
is a superset-free drop-in for it.

For argument conventions (Fortran column-major, `uplo`/`trans`/`diag`
character flags, leading dimensions), consult a standard BLAS reference — the
signatures are unchanged.
