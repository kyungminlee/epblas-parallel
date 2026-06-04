# openblas/kind16 is a faithful hand-port of openblas/kind10

`epblas-openblas` shipped `kind10` only (OpenBLAS D/Z reference kernels
retyped to `long double` / `_Complex long double`). Extending the
reference library to `kind16` (`__float128`) forced two decisions about
*how* the second precision tree comes into being and how it relates to
the first.

## Decision

1. **Hand-port, not generated or build-time-templated.** `kind16` is a
   one-time manual copy-and-edit of every `kind10/*.c`, committed as
   ordinary source. There is no committed generator script and no
   single templated source compiled twice.
2. **Faithful clone, not a fresh implementation.** Each `kind16` routine
   mirrors its `kind10` twin line-for-line — same algorithm, same
   OpenMP block-partitioning, same thresholds — differing only by the
   mechanical quad substitutions (below). This is the *opposite* of the
   production overlay, where `src/epblas-parallel/kind16/` deliberately
   **diverged** from `kind10` (simpler, `int`-indexed, no OMP).

The mechanical substitution set is closed and small:

- `long double` → `__float128`; `_Complex long double` → `__complex128`
  (GCC rejects `_Complex __float128`; `__complex128` comes from
  `<quadmath.h>`).
- Real literal suffix `…L` → `…Q`. Every imaginary literal in `kind10`
  is zero-valued (`0.0L + 0.0iL`, `1.0L + 0.0iL`), so they become
  `(__complex128)0` / `(__complex128)1` — no `__builtin_complex`, no
  non-zero imaginary constants exist.
- libm: `sqrtl→sqrtq`, `ldexpl→ldexpq`, `conjl→conjq`, `creall→crealq`,
  local `cconjl→cconjq` (body uses type-generic `__real__`/`__imag__`).
  `fabsl → __builtin_fabsf128` (single instruction, preserving `fabsl`'s
  single-`fabs` intent; **not** the slower `fabsq` libcall). Plain
  double `ceil` is left as-is, as in `kind10`.
- `<math.h>` → `<quadmath.h>`. Symbols: `e→q`, `y→x`, `ie→iq`, `iy→ix`,
  `ey→qx`, `ye→xq`.

## Why

- **Hand-port over generator.** A re-runnable generator (committed output
  + transform script) would guarantee identity and survive `kind10`
  churn, but it adds a build/maintenance artifact for a *reference*
  library that changes rarely. The owner chose the simpler artifact set:
  plain source, no script to keep correct. The cost — manual re-sync when
  `kind10` changes — is accepted and recorded here so it is not a
  surprise.
- **Hand-port over build-time template.** A single macro-templated source
  is the DRYest option, but it would require retrofitting the *working*
  `kind10` tree with `#ifdef`-quad branches (imaginary literals, conj,
  libm names, headers) and breaks the repo-wide one-directory-per-kind
  convention. Rejected as too invasive for the upside.
- **Faithful over divergent.** The reference library exists for A/B
  comparison against the production overlay and the migrated baseline; a
  faithful retype keeps the comparison honest (same algorithm at two
  precisions) and makes correctness a near-corollary of the already-
  tested `kind10`. The production overlay diverges because it is tuned
  per precision; the reference deliberately does not.

## Consequences

- **`kind10` and `kind16` must be kept in sync by hand.** A change to a
  `kind10` reference kernel that should also apply to `kind16` is two
  edits, not one. A future maintainer who expects a generator will not
  find one — this is intentional.
- `epblas-openblas` now ships `epblas-openblas::qblas` alongside `eblas`;
  discovery is still src-dir-existence + `find_package(qblas)` for tests.
- Consistency tests gain a `target_kind16` shim set (`rk_value = 16`,
  q/x routine names) reusing the shared `tests/epblas-parallel/` body
  templates; they require an installed `eplinalg::qblas` baseline.
- If the two trees drift in practice, the escape hatch is to promote the
  ad-hoc port into a generator later — at the cost of a one-time
  reconciliation of whatever drift accumulated.
