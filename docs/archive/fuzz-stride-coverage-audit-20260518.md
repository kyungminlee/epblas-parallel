# Fuzz / perf stride coverage audit — 2026-05-18

> **Pre-split paths.** Written when the parallel-BLAS overlay lived inside the
> `eplinalg` repo. Translate references as follows:
> - `src/parallel_blas/{kind10,kind16,multifloats}/` → `src/epblas-parallel/{kind10,kind16,multifloats}/`
> - `tests/blas_parallel/{consistency,bench,perf}/` → `tests/epblas-parallel/{consistency,bench,perf}/`
> - `/tmp/stage-{e,q,m}/blas/src/` → `blas/src/` in the sibling `eplinalg` repo (migrated Fortran reference)
> - `/tmp/stage-{e,q,m}/parallel_blas/` → `src/epblas-parallel/...` here
> - `doc/parallel-blas-optimization-findings-20260515.md` and any "Rule N / Addendum N" citations → `doc/optimization-findings.md` here.

Triggered by the loop-direction survey work — when we extended
`fuzz_trsv/trmv` to randomize incx and re-ran, the existing fuzz
had never exercised the general-stride fallback paths in the
overlay. Same risk profile applies to other L2 routines.

## TL;DR

| Class of parameter | Status |
|---|---|
| `lda` (all routines) | ✅ randomized via `n + rand_pad()` |
| `lda` for banded — gbmv/hbmv/sbmv/tbmv/tbsv | ✅ `(k+1) + rand_pad()` or `(kl+ku+1) + rand_pad()` |
| `ldb` / `ldc` in L3 (gemm, hemm, her2k, herk, symm, syr2k, syrk, trmm, trsm, gemmtr) | ✅ independent `rand_pad()` for each |
| `incx` / `incy` in L1 (axpy, copyswap, dot, iamax, norm, rot, rotm, scal) | ✅ `rand_inc()` / `pick_inc()` returning {1, 2, -1, -2} |
| `incx` / `incy` in **trsv / trmv** | ✅ fixed 2026-05-18 (commit `3ba3f485`) |
| `incx` / `incy` in **rest of L2** | ❌ **hardcoded to 1** |

## L2 routines with hardcoded incx=incy=1 in fuzz

Per-fypp grep of `tests/blas_parallel/consistency/`:

| Fuzz body | Has `incx` param | Has `incy` param | Uses `rand_inc`? |
|---|---|---|---|
| `fuzz_gbmv_{real,complex}_body.fypp` | yes | yes | no |
| `fuzz_gemv_{real,complex}_body.fypp` | yes | yes | no |
| `fuzz_ger_{real,complex}_body.fypp` | yes | yes | no |
| `fuzz_hbmv_body.fypp` | yes | yes | no |
| `fuzz_hemv_body.fypp` | yes | yes | no |
| `fuzz_her_body.fypp` | yes | — | no |
| `fuzz_her2_body.fypp` | yes | yes | no |
| `fuzz_hpmv_body.fypp` | yes | yes | no |
| `fuzz_hpr_body.fypp` | yes | — | no |
| `fuzz_hpr2_body.fypp` | yes | yes | no |
| `fuzz_sbmv_body.fypp` | yes | yes | no |
| `fuzz_spmv_body.fypp` | yes | yes | no |
| `fuzz_spr_body.fypp` | yes | — | no |
| `fuzz_spr2_body.fypp` | yes | yes | no |
| `fuzz_symv_real_body.fypp` | yes | yes | no |
| `fuzz_syr_real_body.fypp` | yes | — | no |
| `fuzz_syr2_real_body.fypp` | yes | yes | no |
| `fuzz_tbmv_{real,complex}_body.fypp` | yes | — | no |
| `fuzz_tbsv_{real,complex}_body.fypp` | yes | — | no |
| `fuzz_tpmv_{real,complex}_body.fypp` | yes | — | no |
| `fuzz_tpsv_{real,complex}_body.fypp` | yes | — | no |

That's **21 routine families × 1-3 fypp variants each ≈ 30 fypp files**
where the fuzz hardcodes incx=incy=1 even though the overlay
implements general-stride fallbacks.

What this means concretely: a correctness bug in the `incx ≠ 1`
branch of any of these routines (off-by-one in the strided index,
wrong direction of stride accumulation, sign error when `incx < 0`)
would pass fuzz with 200/200 zero-error indefinitely. The
loop-direction survey's Sub-class A in `ytrsv.c` was exactly this
class of bug — undetected until incx≠1 was exercised.

## Same gap in the perf harness

Per `scripts/gen_perf_harnesses.py`, only `emit_trmv_trsv` was
extended to loop over `BLAS_PERF_INCX` (this session, commit
`4722f49e`). The other L2 emitters still pass a literal `&one`:

| Emitter (gen_perf_harnesses.py) | Routines | Strides covered |
|---|---|---|
| `emit_trmv_trsv` | trsv, trmv | ✅ `BLAS_PERF_INCX` default `{1, 2}` |
| `emit_gemv` | gemv | ❌ incx=incy=1 only |
| `emit_gbmv` | gbmv | ❌ incx=incy=1 only |
| `emit_ger` | ger, gerc, geru | ❌ incx=incy=1 only |
| `emit_symv_hemv` | symv, hemv | ❌ incx=incy=1 only |
| `emit_syr_her` | syr, her | ❌ incx=1 only |
| `emit_spr_hpr` | spr, hpr | ❌ incx=1 only |
| `emit_spmv_hpmv` | spmv, hpmv | ❌ incx=incy=1 only |
| `emit_sbmv_hbmv` | sbmv, hbmv | ❌ incx=incy=1 only |
| `emit_tbmv_tbsv` | tbmv, tbsv | ❌ incx=1 only |
| `emit_tpmv_tpsv` | tpmv, tpsv | ❌ incx=1 only |

Implication: even if correctness in `incx ≠ 1` paths is OK, a
Sub-class-A-style perf cliff (forward-inner under descending-outer
collapsing to 0.30× when out-of-cache) could be hiding in any of
these and would never show up on bench reports. The loop-direction
survey's `ytrsv LCN/x2 N=512: 0.301×` finding was invisible until
the perf harness was extended.

## Stride helper inconsistency

The existing per-fypp helpers diverge:

- `rand_inc()` (axpy_{real,complex}, ger_{real,complex} when added):
  returns {1, 2, -1, -2}, 50/25/15/10
- `pick_inc()` (copyswap_{real,complex}, dot_{real,complex}, iamax_*,
  norm_*, rot_*, scal_*): returns {1, 2, -1, -2}, same distribution
  but defined per-fypp
- `rand_incx()` (added this session in `common/fuzz_util_body.fypp`):
  returns {1, 2, 3, -1, -2}, 50/20/15/10/5

These should be unified. The right home is
`tests/blas_parallel/common/fuzz_util_body.fypp` (already has
`rand_incx`); the per-fypp duplicates can be removed.

## What's already correct

For the avoidance of doubt: `lda` is *not* under-fuzzed anywhere I
checked. Every fypp that declares an `lda` uses `n + rand_pad()` (or
the banded equivalent), where `rand_pad()` returns:

- 60%: `rand_int_log(0, 4)`
- 30%: `rand_int_log(5, 32)`
- 10%: `rand_int_log(33, 128)`

So lda padding is exercised across a wide range. L3 routines
independently pad `ldb` and `ldc`, so packed-vs-strided buffer-aliasing
bugs would surface.

## Recommended fix workflow

Mirror what was done for trsv/trmv:

1. **Unify the stride helper.** Keep `rand_incx()` in
   `common/fuzz_util_body.fypp`. Remove the per-fypp `rand_inc()` /
   `pick_inc()` definitions and replace call sites.
2. **Extend each L2 fuzz body** to pick `incx = rand_incx()` (and
   `incy = rand_incx()` if applicable), allocate the relevant
   buffers with `1 + (n-1)*abs(incx)` length, plumb the stride
   through the `call ${routine}$(...)` lines, and update the FAIL
   print to include `incx` (and `incy`).
3. **Sanity-test each extended fuzz** by deliberately introducing a
   bug into one `incx ≠ 1` path and confirming the fuzz reports
   non-zero failures (as I did for `etrsv.c:130` this session). Then
   revert.
4. **Extend each L2 perf emitter** in `gen_perf_harnesses.py` to
   loop over `BLAS_PERF_INCX` (default `{1, 2}`), emit `/x2` keys
   for non-unit strides, and pass the chosen stride into
   `${routine}$_` / `${routine}$_migrated_`. Use
   `perf_parse_int_list("BLAS_PERF_INCX", ...)` (added this session
   to `perf_common.h`).
5. **Regenerate all affected perf C files** with
   `uv run scripts/gen_perf_harnesses.py`.
6. **Rebuild + re-run** fuzz/perf across all three targets
   (kind10/kind16/multifloats) and chase any sub-parities that
   show up.

Estimated scope: ~30 fypp files + ~10 `emit_*` functions + a generator
regen touching ~150 perf C files. Mechanical work that would benefit
from being done as a single sweep so the audit table at the top of
this doc collapses to all-✅.

## Findings worth keeping if/when this is done

Any sub-parity uncovered by the extended fuzz/perf is high-signal —
it's a bug the previous CI never caught. Catalog it in
`doc/parallel-blas-optimization-findings-20260515.md` with the same
"Addendum N" + "Rule N" pattern; the loop-direction survey is a good
template for how to write it up.

If kind16 / multifloats again show the empirical-vs-doc-prediction
flip (forward-inner faster despite Fortran walking backward, per
the refined Rule 21 in
`doc/loop-direction-survey-20260518.md`), don't keep applying the
fix blindly across the L2 surface — confirm with measurement first.

## What this audit does *not* cover

- **lda alignment**: `rand_pad()` returns 0-128, but does it stress
  cache-line and SIMD-vector boundary alignment specifically? Some
  bugs (alignment-sensitive SSE loads) only fire when lda is
  misaligned at a particular boundary. Out of scope here.
- **alpha / beta scalar parameters**: most L2/L3 fuzz uses
  `rand_alpha_beta()` which already samples ±1, 0, and random
  values. Not audited in detail.
- **Edge cases**: N=0, N=1, n=k for banded, M=N=0 for level-3 —
  the existing `rand_int_log(1, ...)` floor at 1 means N=0 is not
  fuzzed. Separate concern, separate audit.
- **Mixed-stride buffer aliasing**: e.g. gemv with `x` and `y`
  overlapping. BLAS spec says undefined behavior in those cases,
  but worth confirming the overlay doesn't silently corrupt.
