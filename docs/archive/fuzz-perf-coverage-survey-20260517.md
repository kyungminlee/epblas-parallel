# Fuzz / perf coverage survey — 2026-05-17

> **Pre-split paths.** Written when the parallel-BLAS overlay lived inside the
> `eplinalg` repo. Translate references as follows:
> - `src/parallel_blas/{kind10,kind16,multifloats}/` → `src/epblas-parallel/{kind10,kind16,multifloats}/`
> - `tests/blas_parallel/{consistency,bench,perf}/` → `tests/epblas-parallel/{consistency,bench,perf}/`
> - `/tmp/stage-{e,q,m}/blas/src/` → `blas/src/` in the sibling `eplinalg` repo (migrated Fortran reference)
> - `/tmp/stage-{e,q,m}/parallel_blas/` → `src/epblas-parallel/...` here
> - `doc/parallel-blas-optimization-findings-20260515.md` and any "Rule N / Addendum N" citations → `docs/optimization-findings.md` here.

> **Status (2026-05-18):** all six gaps documented below have been
> closed. The tables and gap entries below are kept for historical
> reference; the inline `✅ DONE (<short-sha>)` annotations mark the
> commits that landed each fix. See the "Status update — 2026-05-18"
> section near the bottom for a fresh summary.

Follow-up to `doc/fuzz-stride-coverage-audit-20260518.md`. That audit
covered stride coverage only; this one expands the lens to all
categorical BLAS parameters (`trans`, `side`, `uplo`, `diag`) and
re-checks the stride gap.

Same risk shape as before: a code path that nothing exercises will
silently regress (correctness or perf). The loop-direction survey's
`ytrsv` Sub-class A bug is the canonical example.

## TL;DR

> All bullets below were true at write time (2026-05-17). All have
> since been closed — see commits referenced in the gap list.

- **Stride gap from 2026-05-18 is still wide open.** Only kind10/kind16
  trsv/trmv (the shared `_body.fypp`) were fixed. 20 other L2 routine
  families still hardcode `incx=incy=1` in both fuzz and perf.
- **Multifloats fuzz was *not* included in the 2026-05-18 fix.**
  `target_multifloats/fuzz_m{trsv,trmv}.fypp` are standalone files
  (not body-include shims like kind10/kind16) and still pass `incx=1`.
- **DIAG='U' is never benched** for any triangular routine
  (trmv/trsv/tbmv/tbsv/tpmv/tpsv/trmm/trsm). The unit-diagonal path
  has its own arithmetic shape; a perf regression there is invisible.
- **GEMMTR perf samples only 3 of 9 (ta,tb) pairs.** TT, CN, NC, CT,
  TC, CC are unbenched. **GEMM complex** samples 6 of 9 (skips
  CT/TC/CC).
- **Real L2 triangular fuzz never passes `trans='C'`.** trmv/trsv/
  tbmv/tbsv/tpmv/tpsv real bodies use inline 50/50 N/T; the L3 real
  family (gemm/gemmtr/trmm/trsm) uses the common `rand_trans()`
  helper which emits N/T/C (with `'C'` semantically equal to `'T'`
  for real, but exercising any kernel branch on the literal).
- **Stride helper still fragmented.** `rand_incx`, `rand_inc`,
  `pick_inc`/`pick_inc_pos` coexist; the 2026-05-18 cleanup
  recommendation hasn't landed.

## Fuzz consistency — categorical coverage

Two code paths to audit:
- **Shared bodies** `consistency/fuzz_*_body.fypp` — used by kind10
  (`target_kind10/fuzz_e*.fypp`) and kind16 (`target_kind16/fuzz_q*.fypp`)
  via 2-line `#:include` shims.
- **Multifloats per-target** `consistency/target_multifloats/fuzz_m*.fypp`
  and `fuzz_iw*.fypp` / `fuzz_im*.fypp` — standalone programs, NOT
  shims (each ~65-100 lines with its own randomization logic).

Columns: ✅ randomized, ❌ hardcoded, — n/a.

### Shared bodies (kind10 + kind16 targets)

| Layer | Routine family | trans | side | uplo | diag | strides |
|---|---|---|---|---|---|---|
| L1 | axpy, copyswap, dot, rot, rotm, scal | — | — | — | — | ✅ (±1, ±2) |
| L1 | iamax, norm | — | — | — | — | ✅ pos only |
| L2 | gemv (R/C) | ✅ N/T/C (R now via `rand_trans()`) | — | — | — | ✅ (c797f466) |
| L2 | gbmv (R/C) | ✅ N/T/C | — | — | — | ✅ (c797f466) |
| L2 | ger, gerc, geru | — | — | — | — | ✅ (c797f466) |
| L2 | symv (R), hemv | — | — | ✅ | — | ✅ (c797f466) |
| L2 | sbmv, hbmv | — | — | ✅ | — | ✅ (c797f466) |
| L2 | spmv, hpmv | — | — | ✅ | — | ✅ (c797f466) |
| L2 | syr, her, syr2, her2, spr, hpr, spr2, hpr2 | — | — | ✅ | — | ✅ (c797f466) |
| L2 | trmv, trsv (R/C) | ✅ N/T/C (R via `rand_trans()` 0af68f5f) | — | ✅ | ✅ | ✅ (05-18) |
| L2 | tbmv, tbsv, tpmv, tpsv (R/C) | ✅ N/T/C (R via `rand_trans()` 0af68f5f) | — | ✅ | ✅ | ✅ (c797f466) |
| L3 | gemm (R/C) | ✅ N/T/C (both R and C — `rand_trans()`) | — | — | — | n/a |
| L3 | gemmtr (R/C) | ✅ ta,tb N/T/C | — | ✅ | — | n/a |
| L3 | symm, hemm | — | ✅ | ✅ | — | n/a |
| L3 | syrk, syr2k (R) | ✅ N/T | — | ✅ | — | n/a |
| L3 | syrk, syr2k (C) | ✅ N/T (no 'C' — invalid for complex syrk per BLAS) | — | ✅ | — | n/a |
| L3 | herk, her2k | ✅ N/C | — | ✅ | — | n/a |
| L3 | trmm, trsm (R/C) | ✅ N/T/C (`rand_trans()`) | ✅ | ✅ | ✅ (30/70 U/N) | n/a |

### Multifloats per-target (separate code, not body-include)

Two parallel families — `fuzz_m*` (real DD, `real64x2`) and `fuzz_w*`
(complex DD, `cmplx64x2`). Both are standalone programs.

| Layer | Routine family | trans | side | uplo | diag | strides |
|---|---|---|---|---|---|---|
| L1 | maxpy, mcopy, mdot, mrot, mrotm, mscal, mswap | — | — | — | — | ✅ (±1, ±2) |
| L1 | masum, mnrm2, mwasum, mwnrm2, imamax, iwamax | — | — | — | — | ✅ pos only |
| L1 | mcabs1, mrotg, mrotmg | — | — | — | — | — |
| L1 | waxpy, wcopy, wdotc, wdotu, wmrot, wmscal, wrotg, wscal, wswap | — | — | — | — | ✅ (9be2ebba — unified to `rand_incx[_pos]()`) |
| L2 | mgemv | ✅ N/T | — | — | — | ✅ (70dcfbd8) |
| L2 | mgbmv | ✅ N/T | — | — | — | ✅ (70dcfbd8) |
| L2 | mger | — | — | — | — | ✅ (70dcfbd8) |
| L2 | msymv | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | msbmv | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | mspmv | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | msyr, msyr2, mspr, mspr2 | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | mtrmv, mtrsv | ✅ N/T | — | ✅ | ✅ | ✅ (70dcfbd8) |
| L2 | mtbmv, mtbsv, mtpmv, mtpsv | ✅ N/T | — | ✅ | ✅ | ✅ (70dcfbd8) |
| L2 | wgemv | ✅ N/T/C | — | — | — | ✅ (70dcfbd8) |
| L2 | wgbmv | ✅ N/T/C | — | — | — | ✅ (70dcfbd8) |
| L2 | wgerc, wgeru | — | — | — | — | ✅ (70dcfbd8) |
| L2 | whemv | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | whbmv | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | whpmv | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | wher, wher2, whpr, whpr2 | — | — | ✅ | — | ✅ (70dcfbd8) |
| L2 | wtrmv, wtrsv | ✅ N/T/C | — | ✅ | ✅ | ✅ (70dcfbd8) |
| L2 | wtbmv, wtbsv, wtpmv, wtpsv | ✅ N/T/C | — | ✅ | ✅ | ✅ (70dcfbd8) |
| L3 | mgemm | ✅ ta,tb (via `rand_trans()` → N/T/C) | — | — | — | n/a |
| L3 | mgemmtr | ✅ ta,tb | — | ✅ | — | n/a |
| L3 | msymm | — | ✅ | ✅ | — | n/a |
| L3 | msyrk, msyr2k | ✅ N/T | — | ✅ | — | n/a |
| L3 | mtrmm, mtrsm | ✅ N/T/C (`rand_trans()`) | ✅ | ✅ | ✅ (30/70) | n/a |
| L3 | wgemm, wgemmtr, wsymm, whemm, wsyrk, wsyr2k, wherk, wher2k, wtrmm, wtrsm | ✅ (mirrors `m*`) | (per routine) | ✅ | (trmm/trsm only) | n/a |

Common helpers in `common/fuzz_util_body.fypp`: `rand_trans()` →
N/T/C ~1/3 each; `rand_trans_complex()` is just an alias; `rand_incx()`
→ {1,2,3,-1,-2} (added 05-18). No `rand_uplo`/`rand_side`/`rand_diag`
helpers exist — each fypp inlines its own 50/50 (or 30/70 for trmm/
trsm diag).

## Perf harness — categorical coverage

`scripts/gen_perf_harnesses.py` and generated harnesses under
`tests/blas_parallel/perf/target_*/`.

| Emitter | Routines | trans | side | uplo | diag | strides |
|---|---|---|---|---|---|---|
| emit_gemv | gemv | ✅ | — | — | — | ✅ INCX×INCY cross product, neg in default (ad193439) |
| emit_ger | ger, gerc, geru | — | — | — | — | ✅ INCX×INCY (ad193439) |
| emit_gbmv | gbmv | ✅ | — | — | — | ✅ INCX×INCY (ad193439) |
| emit_symv_hemv | symv, hemv | — | — | ✅ | — | ✅ INCX×INCY (ad193439) |
| emit_sbmv_hbmv | sbmv, hbmv | — | — | ✅ | — | ✅ INCX×INCY (ad193439) |
| emit_spmv_hpmv | spmv, hpmv | — | — | ✅ | — | ✅ INCX×INCY (ad193439) |
| emit_syr_her | syr, her | — | — | ✅ | — | ✅ INCX incl. neg (ad193439) |
| emit_spr_hpr | spr, hpr | — | — | ✅ | — | ✅ INCX incl. neg (ad193439) |
| emit_trmv_trsv | trmv, trsv | ✅ | — | ✅ | ✅ N/U (249e1a01) | ✅ INCX incl. neg |
| emit_tpmv_tpsv | tpmv, tpsv | ✅ | — | ✅ | ✅ N/U (249e1a01) | ✅ INCX incl. neg |
| emit_tbmv_tbsv | tbmv, tbsv | ✅ | — | ✅ | ✅ N/U (249e1a01) | ✅ INCX incl. neg |
| emit_gemm | gemm | ✅ R: {NN,TN,NT,TT}; C: 6 of 9 (skips CT/TC/CC) | — | — | — | n/a |
| emit_symm_hemm | symm, hemm | — | ✅ | ✅ | — | n/a |
| emit_syrk_herk | syrk, herk | ✅ | — | ✅ | — | n/a |
| emit_trmm_trsm | trmm, trsm | ✅ N/T/C | ✅ | ✅ | ✅ N/U (249e1a01) | n/a |
| emit_gemmtr | gemmtr | ✅ full grid: 4 (R) / 9 (C) (ta,tb) (249e1a01) | — | ✅ | — | n/a |

Note: `emit_gemm` complex still samples 6 of 9 trans pairs (skips
CT/TC/CC). That subgap was not in the original 6 gaps and remains
open — see "Status update — 2026-05-18" below.

## Gaps, ranked by risk

1. ✅ **CLOSED (c797f466 + 70dcfbd8).** L2 stride coverage —
   28 shared-body fypps (c797f466) and the 33 standalone multifloats
   L2 fypps (16 `m*` + 17 `w*`, 70dcfbd8) all now randomize
   `incx`/`incy` via `rand_incx()` with allocation
   `1 + (n-1)*abs(inc)`.
2. ✅ **CLOSED (249e1a01).** DIAG='U' now benched: a
   `diags[] = {'N', 'U'}` loop landed in `emit_trmv_trsv`,
   `emit_tpmv_tpsv`, `emit_tbmv_tbsv`, and `emit_trmm_trsm`. Key
   format already encoded diag so existing parsers consumed the new
   variants unchanged.
3. ✅ **CLOSED (249e1a01).** `emit_gemmtr` now samples the full
   `(ta, tb)` grid — 4 pairs for real (N/T × N/T), 9 for complex
   (N/T/C × N/T/C). The earlier 3-pair sample masked a 0.03x
   `egemmtr UNN` sub-parity which is now visible.
4. ✅ **CLOSED (c797f466 fuzz + ad193439 perf).** GER family
   randomizes both strides in fuzz, and `emit_ger` now sweeps
   `BLAS_PERF_INCX × BLAS_PERF_INCY` (default including negatives).
5. ✅ **CLOSED (0af68f5f).** Real L2 triangular fuzz now uses
   `rand_trans()` — `'C'` literal branch exercised. trmv/trsv/tbmv/
   tbsv/tpmv/tpsv real bodies all swapped.
6. ✅ **CLOSED (9be2ebba).** All call sites of legacy `pick_inc` /
   `rand_inc` / `pick_inc_pos` redirected to `rand_incx()` (or
   `rand_incx_pos()` for routines where BLAS spec forbids negative
   stride — iamax, norm, scal). Local helper definitions removed; the
   canonical helpers live in `common/fuzz_util_body.fypp` and
   `common/target_multifloats/fuzz_util.fypp`.
   `rand_uplo`/`rand_side`/`rand_diag` were not promoted — still
   inlined per fypp.

## Recommended sweep

Mirror the trsv/trmv playbook from 2026-05-18 in one bulk pass so
the audit/survey tables collapse to all-✅:

1. **Plumb `rand_incx()` / `rand_incy()` into all remaining L2 fuzz
   bodies AND the standalone multifloats L2 fuzz files (both `m*`
   real-DD and `w*` complex-DD).** Allocate buffers as
   `1 + (n-1)*abs(inc)`; thread strides through both kernel calls;
   add stride values to FAIL prints. Per the 05-18 workflow,
   sanity-test each by injecting a bug into one `incx ≠ 1` path and
   confirming non-zero failures, then revert.
2. **Extend the 10 affected `emit_*` functions** in
   `gen_perf_harnesses.py` to loop `BLAS_PERF_INCX` (and
   `BLAS_PERF_INCY` where applicable). Emit `/xK` key suffixes for
   non-unit strides. Use `perf_parse_int_list` (already added in
   `perf_common.h`).
3. **Add `BLAS_PERF_DIAG` loop** (default `{N, U}`) to
   `emit_trmv_trsv`, `emit_tpmv_tpsv`, `emit_tbmv_tbsv`,
   `emit_trmm_trsm`. Encode diag into the key (e.g. `ULU` vs `ULN`).
4. **Extend `emit_gemmtr` (ta,tb) sample set** to all 9 pairs. The
   loop-direction survey shows trans choice flips the inner walk
   direction; cheap to cover fully.
5. **Unify stride helper.** Keep `rand_incx()` in
   `common/fuzz_util_body.fypp`. Delete per-fypp `rand_inc()` /
   `pick_inc()` / `pick_inc_pos()`; redirect call sites. Add
   `rand_uplo`/`rand_side`/`rand_diag` while at it.
6. **Switch L2 real triangular bodies to `rand_trans()`** so 'C' is
   covered (one-line per file).
7. **Regenerate** all perf C files via
   `uv run scripts/gen_perf_harnesses.py`.
8. **Rebuild + re-run** fuzz/perf across kind10/kind16/multifloats.
   Catalog any sub-parities as new Addenda in
   `doc/parallel-blas-optimization-findings-20260515.md`, mirroring
   the loop-direction survey's Addendum + Rule format.

Estimated scope: ~28 shared-body fypps (done 2026-05-18 follow-up
session) + ~33 multifloats fypps (16 m* + 17 w*) + ~10 perf emitters
+ ~150 perf C regen. Single sweep; the survey tables collapse on
completion.

## Status update — 2026-05-18

All six gaps above were closed in a single follow-up session. Commits
on `parallel-blas`:

- `0af68f5f` — 6 real L2 triangular fuzz bodies → `rand_trans()`
- `70dcfbd8` — 33 multifloats L2 fuzz drivers (16 `m*` + 17 `w*`) →
  randomize `incx`/`incy`; `rand_incx()` added to multifloats
  `fuzz_util`
- `9be2ebba` — 31 fypp files unified to `rand_incx[_pos]()`; legacy
  `pick_inc`/`rand_inc`/`pick_inc_pos` removed; scal routes through
  `rand_incx_pos()` per BLAS spec
- `249e1a01` — `gen_perf_harnesses.py`: DIAG={N,U} loop on 4
  triangular emitters; `emit_gemmtr` full (ta,tb) grid
- `ad193439` — `gen_perf_harnesses.py`: independent `BLAS_PERF_INCY`
  env var (defaults to mirror INCX); default stride set bumped to
  `{1, 2, -1}` so negative-stride code paths are benched by default
  too
- Plus regen commits (c7e2d0e4, 38b3c921, a82f0370) for the 195
  perf C/C++ files.

Smoke verification on the new defaults exposed two latent
sub-parities that the old narrow sweep masked:

- `egemmtr UNN` ≈ 0.031x (overlay ~32× slower than migrated)
- `etrsv LNN /x2` ≈ 0.054x (overlay ~19× slower than migrated at
  L/N/N with stride=2)

Both are follow-up work, not addressed in this sweep.

### Gaps that remain (out of scope for the 2026-05-17 survey)

- **`emit_gemm` complex still samples 6 of 9 trans pairs** (skips
  CT/TC/CC). Same shape of gap as GEMMTR was, but at row 130 of the
  perf table — not flagged in the original 6 ranked gaps.
- **`rand_uplo`/`rand_side`/`rand_diag` helpers** not promoted to
  `common/`. Every fypp still inlines its own 50/50 (or 30/70 for
  trmm/trsm `diag`). Cosmetic; functionally equivalent.
- The follow-up sub-parities above (egemmtr UNN, etrsv LNN /x2) are
  signals, not gaps in the harness itself.

## What this survey does *not* cover

- **`alpha` / `beta` scalar values.** Most fuzz bodies use
  `rand_alpha_beta()` (samples ±1, 0, random). Not separately
  audited.
- **Edge sizes.** N=0, N=1, M=0, n=k for banded. `rand_int_log(1,...)`
  floors at 1, so N=0 is not fuzzed. Separate concern (same as 05-18).
- **lda alignment.** `rand_pad()` covers 0-128 padding; whether it
  stresses cache-line/SIMD boundaries specifically is out of scope.
- **Buffer aliasing.** Mixed-stride overlap (e.g. gemv with x and y
  pointing into the same allocation) — BLAS spec says undefined, but
  worth confirming no silent corruption. Out of scope.
