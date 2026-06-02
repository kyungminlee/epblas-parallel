# archive/

Code that is no longer part of the active build, kept for reference and
possible later resurrection. Nothing here is compiled, tested, or wired into
CMake — the directory layout under `archive/` mirrors the original tree so
relative paths between archived files still resolve if the harness is restored.

## Contents

### Legacy Fortran bench harness (archived 2026-06-02)

- `tests/epblas-parallel/bench/` — `bench_*_body.fypp` bench bodies plus the
  per-target shims (`target_kind10/`, `target_kind16/`, `target_multifloats/`).
- `tests/epblas-openblas/bench/` — the epblas-openblas shims (`target_kind10/`),
  which `#include` the bodies from `../epblas-parallel/bench`.

These compiled into `bench_<routine>` / `ep_bench_<routine>` executables that
reported **GFLOP/s** (subject vs migrated). They were superseded by the
kernel-isolated **C perf harness** (`tests/epblas-parallel/perf/`), which is the
supported timing path and reports **bare wall time (ns/call)** — see
`reports/cmp5/` and `doc/optimization-findings.md` → "Reporting convention".

The bench wiring was removed from `tests/epblas-parallel/CMakeLists.txt` and
`tests/epblas-openblas/CMakeLists.txt` (the consistency/fuzz and perf blocks in
those files are unaffected).

### Bench-harness consumer scripts

- `scripts/blas_autotune.py` — block-size autotuner; ran `bench_<r>gemm` over an
  MC/KC/NC grid and selected the best GFLOP/s. Depends on the archived bench
  JSON (`gflops_subject`).
- `scripts/blas_overlay_report.py` — generated `reports/overlay-coverage.md` from
  the archived bench binaries' GFLOP/s stdout.

## Restoring

Move the relevant subtree back to its original path and re-add the corresponding
`# Bench tests` block to the CMakeLists it came from (recoverable from git
history of those two files). The bench harness still emits GFLOP/s; converting it
to ns/call was explicitly deferred when it was archived.
