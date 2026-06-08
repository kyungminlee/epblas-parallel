# epblas-parallel performance gap report — all types (2026-06-07)

Snapshot of the remaining performance gaps in the `epblas-parallel` overlay,
across all six element families, after the multifloats double-double OpenBLAS
hand-port landed (commit `0321ebe`) and the first complete 5-way `cmp5` sweep
with every family in the surface.

## Scope & method

- **Source:** `reports/cmp5/cmp5.tsv` — full 5-way sweep, `REPS=3` interleaved
  min-of-3 (per-cell minimum cancels transient contention), 209 routines,
  8702 `(routine, key, size)` cells, pinned via `taskset` (P-core 0 at OMP=1,
  cores 0–3 at OMP=4).
- **Families:** kind10 (`e`=REAL\*10, `y`=COMPLEX\*10), kind16 (`q`=`__float128`,
  `x`=`__complex128`), multifloats (`m`=`float64x2` double-double real,
  `w`=`std::complex<float64x2>`).
- **Units:** bare wall time (ns/call), smaller = faster. The firm bar is
  **par/ob ≤ 1.0 in every cell** — `par` = the optimized parallel overlay,
  `ob` = the faithful OpenBLAS-style reference clone. `par4/par1` is the
  threading factor (≈0.25 = ideal 4-core scaling, ≈1.0 = ran serial).
- **Two reference legs, two meanings.** Every gap below is measured against the
  *ob clone* (a hand-tuned OpenBLAS-style reference — the strict bar). Against
  the *migrated Fortran* (`migrated-serial`, what actually ships), par is
  at-or-ahead **everywhere**, including in all the gap routines. So these are
  "trails the strict reference" gaps, not regressions against the production
  baseline. The `par1/mig` column is cited where the distinction matters.

## Executive summary

| Family | Serial (par1/ob1) | Threaded (par4/ob4) | Status |
|--------|-------------------|---------------------|--------|
| kind10 `e`/`y` | parity-or-better | parity-or-better | **complete** — only by-design/won't-fix residuals |
| kind16 `q`/`x` | parity except L3 GEMM-family | parity (reductions now threaded) | one real gap: L3 kernel ~10–17% |
| multifloats `m`/`w` | parity except argmax | triangular matvec lags | the live frontier |

Correctness is clean: 75/75 consistency tests pass for every family; serial
paths are bit-exact or within tolerance vs migrated.

---

## Genuine remaining gaps (ranked)

### 1. multifloats OMP=4 triangular matvec — *biggest threading gap*

The canonical unit-stride triangular matvec barely threads in the multifloats
overlay, so it loses ground to the ob clone (which threads every shape) as N
grows. Serial is at parity — this is purely a threading-coverage gap.

| routine | key | N | par1/ob1 | par4/ob4 | par4/par1 |
|---------|-----|--:|---------:|---------:|----------:|
| `mtrmv` | LNN | 1024 | 1.03 | **3.18** | 0.90 |
| `mtpmv` | LNN | 1024 | 1.01 | **3.00** | 0.86 |
| `wtrmv` | LNN | 512 | 1.07 | **2.06** | 0.52 |
| `wtpmv` | LNN | 512 | 1.02 | **1.65** | 0.47 |

- **Cause:** the kind10 row-gather threading
  (`project_l2_rowgather_scaling`) does not transfer to the double-double
  triangular matvec — `par4/par1 ≈ 0.86–0.90` for the real `m` variants means
  almost no parallel speedup. Complex `w` threads partially (≈0.5).
- **Context:** par serial is 3.5× faster than migrated (`mtrmv par1/mig 0.28`);
  only the OMP=4 ceiling trails the ob clone.
- **Fix:** port the kind10 disjoint-output-row gather to the DD `*trmv`/`*tpmv`
  kernels (register-resident dot, single `malloc(n)`, one barrier). This is the
  direct analogue of the kind10 band-matvec rollout already completed.

### 2. multifloats serial argmax (`imamax`/`iwamax`) — ~2× serial kernel

Unlike kind10's `ieamax`/`iyamax` (which are *par-faster* than ob at 0.66/0.58),
the multifloats argmax serial kernel runs ~2× the ob clone at **every** size.

| routine | N range | par1/ob1 | par1/mig | par4/par1 (N≥16384) |
|---------|---------|---------:|---------:|--------------------:|
| `imamax` | 64–65536 | 1.88–2.09 | 0.41 | 0.24 (threads) |
| `iwamax` | 64–65536 | 1.70–2.45 | 0.06 | 0.27 (threads) |

- **Cause:** the serial scan kernel, not threading (it threads cleanly 4× at
  large N). The clear standout regression vs the ob reference.
- **Context:** par still crushes migrated (`iwamax par1/mig 0.06`), so this is
  a "doesn't match the tuned reference" gap, not a production regression.
- **Fix:** mirror the kind10 `ieamax`/`iyamax` serial kernel structure into the
  multifloats argmax (branch/compare layout, `fabs` handling).

### 3. kind16 L3 GEMM-family serial kernel — ~10–17%

The parallel/kind16 L3 microkernel is a "divergent simpler port" (it did not
inherit the kind10 blocked-kernel tuning), so it trails the new faithful ob
clone by a steady ~10–17% at **both** OMP=1 and OMP=4 (threading is fine,
`par4/par1 ≈ 0.27` — ideal 4×). It is a serial kernel gap, not a threading one.

| routine | N | par1/ob1 | par4/ob4 | par1/mig |
|---------|--:|---------:|---------:|---------:|
| `qgemm` | 512 | 1.15 | 1.15 | 1.02 |
| `qsymm` | 512 | 1.14 | 1.15 | 1.10 |
| `xgemm` | 256 | 1.15 | 1.16 | 1.00 |
| `xhemm` | 256 | 1.10 | 1.11 | 1.02 |
| `xsymm` | 256 | 1.10 | 1.11 | 1.02 |
| `xherk`/`xsyrk` | 256 | 1.04–1.07 | 1.04–1.07 | ~1.0 |

- **Cause:** parallel/kind16 L3 diverged from the kind10 source (ADR 0002 was
  reversed for parallel/kind16, faithful only for the *ob* leg). The kind16 ob
  clone *is* the faithful kind10-port, so it now sets a stricter bar the
  divergent overlay misses.
- **Context:** `par1/mig ≈ 1.0–1.10` — at parity with the shipped Fortran; the
  gap exists only against the strict ob clone.
- **Fix:** re-port the kind16 L3 GEMM-family onto the kind10 blocked/fused-packer
  substrate (`project_l3_fused_packer_pattern`,
  `project_egemm_done`). Larger effort; lower severity because it is not a
  production regression.

---

## Structural / by-design (not defects)

These show large `par4/ob4` ratios but are correct design choices, documented
and verified — **no action**.

- **L1 RMW stays serial** (`*axpy`/`*scal`/`*copy`/`*swap`/`*rot*`, both
  precisions where applicable): write-bound RMW does not benefit from threading;
  par's serial ties/beats ob's serial. Residual `eswap` (1.13), `yscal` (1.06),
  `yerot` (1.05), `mswap` (1.12), `wswap` (1.06) OMP=4 cells are this — the ob
  clone threads it for no gain. See `project_x87_accumulator_spill` (trigger 4),
  `project_kind10_rmw_cache_threading`.
- **`tbsv` (band tri-solve, all families) is correctly not threaded** — O(N·K)
  work with a K-deep loop-carried recurrence; OpenBLAS does not thread it
  either (`project_tbsv_serial_by_design`).
- **Strided L2 is parity-or-better**, not a gap: e.g. `egemv` strided cells run
  `par4/ob4 ≈ 0.66–0.73` (par wins), `mgemv`/`qgemv` strided ≈ 1.00–1.03. The
  earlier "strided falls back to serial → 3.5×" caveat no longer holds on this
  surface.
- **Dense rank `mger`/`wgerc`/`wgeru` unit-stride WINS big** (`par1/ob1 ≈
  0.16–0.27`, `par4/ob4 ≈ 0.17–0.28`). The misleading per-routine *median*
  `par4/ob4 ≈ 3.7` is strided-key skew — the canonical unit-stride case is a
  large par win.
- **Tiny scalar / x87-floor soft spots** (won't-fix): `mrotmg` 1.28 and
  `mrotm` 1.08 (tiny absolute ns), `wher` 1.07, `xqrot` 1.06; kind10
  `etbsv`/`ytbsv` UPPER and `esbmv`/`yhbmv`/`ytbmv` serial-UPPER ~1.02–1.05 are
  disassembled-clean x87 outer-loop scheduling floors
  (`project_tbsv_serial_by_design`, `project_x87_accumulator_spill`).

## Closed since the last frontier note

- **kind16 L1 reductions are now threaded** (`project_kind16_parallelization`,
  commits `3c2212c`+`b177e98`). The `cmp5_summary.md` boilerplate still says
  "unported", but the data disagrees: `qasum`/`qdot`/`qnrm2`/`iqamax`/`xdotc`/
  `xdotu`/`ixamax`/`qrot` all show `par4/par1 ≈ 0.27` (threading 4×) with
  `par4/ob4 ≈ 0.23–1.01` (parity or par-win). **No gap.**
- All kind10 L2 OMP gaps, rank-update family, band-matvec rollout, GEMM family,
  and reverse-gap routines are complete (`project_l2_omp_gap`,
  `project_kind10_reverse_gap`, `project_egemm_done`).

---

## Recommended priority order

1. **multifloats triangular matvec OMP=4** (`mtrmv`/`mtpmv`/`wtrmv`/`wtpmv`) —
   biggest gap (up to 3.2×), clean fix (port the kind10 row-gather).
2. **multifloats serial argmax** (`imamax`/`iwamax`) — ~2×, isolated serial
   kernel, mirror kind10.
3. **kind16 L3 GEMM-family** (`qgemm`/`qsymm`/`xgemm`/`xhemm`/`xsymm`/`xherk`/
   `xsyrk`) — ~10–17%, larger re-port effort, not a production regression.

Everything else is at parity-or-better, structurally serial by design, or a
won't-fix x87/scalar floor.

> Note: `reports/cmp5/cmp5_summary.md` header text is stale boilerplate
> ("kind10 + kind16", "min-of-5") — `summarize.py` was not updated for the
> multifloats addition or `REPS=3`. The data rows in `cmp5.tsv` are correct.
