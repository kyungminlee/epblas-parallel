# Task #96 kind16 threading — 5-way interleaved regression sweep

Full 5-way cmp5 surface (epopenblas × {omp1,omp4}, parallel × {omp1,omp4},
shared migrated serial reference), interleaved **min-of-3**, one physical core
per thread, `BLAS_PERF_TIME_BUDGET=0.3`. Driver
`run_cmp5_task96.sh`, aggregated by `agg_task96.py` from `cmp5_task96_raw.tsv`.

Bare wall time (ns/call); **smaller = faster**. Ratios: `p1/mig` = par-serial ÷
migrated-serial (serial parity; ~1.0 ⇒ no regression). `p1/ep1` = par-serial ÷
openblas-serial. `p4/ep4` = par-OMP4 ÷ openblas-OMP4 (≤1.0 ⇒ par matches/beats
openblas threaded). `p4/p1` = par self-scaling (≈0.25 ⇒ ~4×).

## Verdict: zero regressions

No par cell is slower than either baseline (openblas same-omp, or migrated
serial) by more than 5% at any size or thread count.

- **Serial parity preserved** — `p1/mig` ∈ [0.993, 1.031] across all 8 routines
  and all sizes; every value inside the 5% gate. The serial paths are
  byte-for-byte unchanged (gated `#ifdef _OPENMP` fast-paths), and the sweep
  confirms it at the wall-clock level.
- **par ≤ openblas serial** — `p1/ep1` ∈ [0.88, 1.03]; the quad norms
  (qnrm2/qxnrm2) are 0.88–0.93, i.e. par is *faster* serially.
- **OMP=4 threading** — `p4/p1` reaches ~0.225–0.28 at large N (≈3.6–4.4×) for
  qnrm2, qxnrm2, qgbmv, xgbmv; ~0.26–0.30 for the rank-updates qsyr/qspr/xher/
  xhpr. par matches or beats the openblas OMP=4 reference (`p4/ep4` ≤ 1.0) at
  every measured cell.

## Highlights (largest measured size per routine)

| routine | size  | p1/mig | p4/p1 | p4/ep4 |
|---------|------:|-------:|------:|-------:|
| qnrm2   | 65536 |  1.010 | 0.250 |  0.225 |
| qxnrm2  | 65536 |  1.021 | 0.250 |  0.232 |
| qgbmv   |  1024 |  1.001 | 0.274 |  0.859 |
| xgbmv   |   512 |  1.004 | 0.277 |  0.913 |
| qsyr    |  1024 |  1.013 | 0.262 |  0.982 |
| qspr    |  1024 |  1.028 | 0.262 |  0.997 |
| xher    |   512 |  1.004 | 0.262 |  0.949 |
| xhpr    |   512 |  1.003 | 0.257 |  0.937 |

(`p4/ep4` near 1.0 on the rank-updates at large N is parity, not a regression:
openblas threads the same triangular update and the kernels converge to the
libquadmath compute ceiling. The decisive numbers are `p4/p1` — par's own
serial→threaded scaling — which is ~4× everywhere.)
