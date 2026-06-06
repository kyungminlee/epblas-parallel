# Parallelization-gap tracker — kind16 & multifloats

Goal: fill the OpenMP parallelization gap so kind16 and multifloats reach the same
threading coverage as kind10. **Fuzz AND bench (5-way cmp5) before AND after every
change** to guard against regressions. Serial paths must stay BIT-EXACT.

## cmp5 columns (filled from a warmed, interleaved min-of-N sweep on an idle machine)

Bare wall time → ratios, **smaller = faster**. Same convention as
`task96_kind16_interleaved.md`. NOTE: multifloats (double-double) has **no
OpenBLAS leg** (DD is not in OpenBLAS), so its bracket is **3-way**
(par-omp1, par-omp4, migrated-serial) — the `ep*` columns do not apply; we track
`mig`-based ratios instead:

- `p1/mig` — par-serial ÷ migrated-serial. **Serial parity / serial-speedup**;
  ≤1.0 ⇒ par serial already ≥ migrated (these DD kernels run ~0.07–0.1, i.e.
  10–15× faster serially). Must not regress vs the before-baseline.
- `p4/mig` — par-OMP4 ÷ migrated-serial (end-to-end win vs the reference).
- `p4/p1` — par self-scaling (≈0.25 ⇒ ~4× on 4 cores; the headline threading number).

(kind16 keeps the full 5-way `ep*` convention since OpenBLAS has kind16 via the
hand-port — but kind16 has no gap, so this tracker only fills multifloats cells.)

Cells are `—` until measured. `status`: `TODO` (gap to fill) · `WIP` · `DONE`
(threaded, parity-or-better, fuzz+bench clean) · `N/A` (by-design serial).

---

## kind16 — COMPLETE (no gap)

67/75 routines threaded (106 parallel regions). The 8 unthreaded are by-design
serial and need no work:

| routine | reason |
|---|---|
| qcabs1, qrotg, qrotmg, xrotg | scalar generators (no loop to thread) |
| qtbsv, qtpsv, xtbsv, xtpsv | loop-carried band/packed triangular solves |

Verified zero-regression (`task96_kind16_interleaved.md`): `p1/mig` ∈ [0.993, 1.031],
`p4/ep4` ≤ 1.0, `p4/p1` ~0.225–0.30. Nothing to do here unless a re-sweep surfaces a
regression.

---

## multifloats — 35 routines to fill

### L1 — real (m / float64x2)

| routine | op | p1/mig | p4/mig | p4/p1 | notes | status |
|---|---|---|---|---|---|---|
| masum  | asum            | 0.061 | 0.015 | 0.253 | thr n>8192; serial ≤8192 untouched, bit-exact | DONE |
| maxpy  | axpy            | 0.067 | 0.017 | 0.259 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| mcopy  | copy            | 0.867 | 0.344 | 0.397 | thr n>8192; BW-bound memcpy split ~2.5×, bit-exact | DONE |
| mdot   | dot             | 0.068 | 0.018 | 0.260 | thr n>8192; serial untouched; reduction within fuzz tol | DONE |
| mnrm2  | nrm2            | — | — | — | — | TODO |
| mrot   | rot             | 0.054 | 0.016 | 0.291 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| mrotm  | rotm            | 0.316 | 0.082 | 0.260 | thr n>1024; scalar parallel-for, bit-exact | DONE |
| mscal  | scal            | 0.193 | 0.067 | 0.346 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| mswap  | swap            | 1.038 | 0.650 | 0.627 | thr n>8192; BW-bound swap ~1.6×, bit-exact | DONE |
| mwasum | asum(|x|, cplx) | 0.007 | 0.002 | 0.266 | thr n>8192; serial untouched, bit-exact | DONE |
| mwnrm2 | nrm2(cplx)      | — | — | — | — | TODO |
| imamax | iamax           | 0.396 | 0.088 | 0.221 | thr n>8192; argmax-merge bit-exact (lowest-index tie) | DONE |

### L1 — complex (w / complex64x2)

| routine | op | p1/mig | p4/mig | p4/p1 | notes | status |
|---|---|---|---|---|---|---|
| waxpy  | axpy            | 0.082 | 0.022 | 0.268 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| wcopy  | copy            | 1.081 | 0.407 | 0.377 | thr n>8192; BW-bound memcpy split ~2.7×, bit-exact | DONE |
| wdotc  | dotc            | 0.080 | 0.021 | 0.263 | thr n>8192; serial untouched; reduction within fuzz tol | DONE |
| wdotu  | dotu            | 0.085 | 0.023 | 0.266 | thr n>8192; serial untouched; reduction within fuzz tol | DONE |
| wmrot  | rot (real cs)   | 0.032 | 0.009 | 0.273 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| wmscal | scal (real a)   | 0.042 | 0.015 | 0.355 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| wscal  | scal (cplx a)   | 0.080 | 0.023 | 0.288 | thr n>2048; disjoint SIMD slices, bit-exact | DONE |
| wswap  | swap            | 0.884 | 0.681 | 0.771 | thr n>8192; BW-bound swap ~1.3×, bit-exact | DONE |
| iwamax | iamax           | 0.048 | 0.012 | 0.257 | thr n>8192; argmax-merge bit-exact (lowest-index tie) | DONE |

### L2 — real (m / float64x2)

| routine | op | p1/mig | p4/mig | p4/p1 | notes | status |
|---|---|---|---|---|---|---|
| msbmv  | sym band mv     | — | — | — | — | TODO |
| mspmv  | sym packed mv   | — | — | — | — | TODO |
| msymv  | sym mv          | — | — | — | — | TODO |
| mtbmv  | tri band mv     | — | — | — | — | TODO |
| mtpmv  | tri packed mv   | — | — | — | — | TODO |
| mtrmv  | tri mv          | — | — | — | — | TODO |
| mtrsv  | tri solve       | — | — | — | — | TODO |

### L2 — complex (w / complex64x2)

| routine | op | p1/mig | p4/mig | p4/p1 | notes | status |
|---|---|---|---|---|---|---|
| whbmv  | herm band mv    | — | — | — | — | TODO |
| whemv  | herm mv         | — | — | — | — | TODO |
| whpmv  | herm packed mv  | — | — | — | — | TODO |
| wtbmv  | tri band mv     | — | — | — | — | TODO |
| wtpmv  | tri packed mv   | — | — | — | — | TODO |
| wtrmv  | tri mv          | — | — | — | — | TODO |
| wtrsv  | tri solve       | — | — | — | — | TODO |

### N/A — by-design serial (no work)

| routine | reason |
|---|---|
| mcabs1, mrotg, mrotmg, wrotg | scalar generators |
| mtbsv, mtpsv, wtbsv, wtpsv   | loop-carried band/packed triangular solves |

### Already threaded (32 — DONE, re-sweep to confirm parity if touched)

L3: mgemm wgemm mgemmtr wgemmtr mtrsm wtrsm mtrmm wtrmm msyrk wsyrk wherk msymm
wsymm whemm msyr2k wsyr2k wher2k.
L2: mgbmv wgbmv mgemv wgemv mger wgerc wgeru mspr wspr(=whpr) mspr2 whpr2 msyr wher
msyr2 wher2.
