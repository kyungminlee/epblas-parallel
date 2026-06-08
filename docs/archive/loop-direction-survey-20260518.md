# Loop-direction survey of the parallel-blas overlay — 2026-05-18

> **Pre-split paths.** Written when the parallel-BLAS overlay lived inside the
> `eplinalg` repo. Translate references as follows:
> - `src/parallel_blas/{kind10,kind16,multifloats}/` → `src/epblas-parallel/{kind10,kind16,multifloats}/`
> - `tests/blas_parallel/{consistency,bench,perf}/` → `tests/epblas-parallel/{consistency,bench,perf}/`
> - `/tmp/stage-{e,q,m}/blas/src/` → `blas/src/` in the sibling `eplinalg` repo (migrated Fortran reference)
> - `/tmp/stage-{e,q,m}/parallel_blas/` → `src/epblas-parallel/...` here
> - `doc/parallel-blas-optimization-findings-20260515.md` and any "Rule N / Addendum N" citations → `docs/optimization-findings.md` here.

Static survey of every routine under `src/parallel_blas/{kind10,kind16,multifloats}/`
for the cache-direction bug class fixed in commit `ac6fe030`
(see Addendum 18 / Rule 21 in
`doc/parallel-blas-optimization-findings-20260515.md`).

## The pattern

A nested loop where the **outer descends** (`for (i = N-1; i >= 0; --i)`)
but the **inner ascends** (`for (k = a; k < b; ++k)`) and the inner
touches a vector (`x[k]` / `y[k]`) that gets re-read at smaller indices
in the next outer iter. Under cache pressure (matrix > L2):
- Forward inner ends each iter at the high end of x; next outer iter
  opens at the low end ⇒ x falls out of L1 and is re-streamed.
- Backward inner ends near where the next iter begins ⇒ x stays hot.

Functionally equivalent, performance-wise not. Measured impact on the
fixed routines: `etrsv LTN` 0.43× → 0.96× at N=1024; `ytrsv LTN/LCN`
0.29× → 1.00× at N=512.

## Methodology

Three parallel subagents (one per target) grepped each `*.c` / `*.cpp`
under the target directory for descending-outer loops, opened each
hit, and cross-checked the corresponding Fortran reference under
`/tmp/stage-{e,q,m}/blas/src/<name>.f`. Verdict per site:
- **likely-affected** — Fortran walks the inner backward; overlay walks
  forward → real bug.
- **matches-Fortran** — Fortran also walks forward at the same site;
  direction is consistent.
- **no-conflict** — descending outer + descending inner, already correct.

## Verification (2026-05-18, after the subagent survey)

Every claim in this document was re-checked by reading the cited
files directly and cross-checking against the migrated Fortran
sources under `/tmp/stage-{e,q,m}/blas/src/`. Findings:

**Sub-class A (2 sites cited):** confirmed.
- `etrsv.c:127-132` — outer `i = N-1..0`, inner `k = i+1..N-1` reading
  `x[kx + k*incx]`. ✓
- `ytrsv.c:118-126` — same shape with combined conj/non-conj inner. ✓

**Sub-class B (4 files cited):** confirmed.
- `qtrsv.c:60,101` — LTN both strides walk forward; Fortran qtrsv.f
  `DO 130 I = N,J+1,-1` (and DO 150) walk backward. ✓
- `xtrsv.c:59,111` — LTN/LCN both strides, both conj branches walk
  forward; Fortran xtrsv.f DO 150/160/180/190 walk backward. ✓
- `mtrsv.cpp:153 (SIMD), 238 (scalar), 280 (stride)` — all walk
  forward; Fortran mtrsv.f DO 130/150 walk backward. ✓
- `wtrsv.cpp:214 (SIMD lambda), 248 (scalar), 301 (stride)` — all walk
  forward; Fortran wtrsv.f DO 150/160/180/190 walk backward. ✓

**Sub-class C (6 files × 2 strides = 12 sites cited):** confirmed.
- All 6 trmv files (`{e,y,q,x,m,w}trmv`) have the NL path with outer
  `j = N-1..0` and inner `i = j+1..N-1` writing `x[i]`, at both
  incx=1 and incx≠1. ✓
- Fortran etrmv.f `DO 50 I = N,J+1,-1` (line 259) and `DO 70` (272)
  walk inner backward. Same in qtrmv/xtrmv/mtrmv/wtrmv references. ✓

**Sub-class D (6 files × 2 strides + conj branches cited):** confirmed.
- All 6 trmv files have UT/UC with outer `j = N-1..0` and inner
  `i = 0..j-1` reading `x[i]`, at both strides; complex variants
  branch on `conj_a` inside. ✓
- Fortran etrmv.f `DO 90 I = J-1,1,-1` (line 291) and `DO 110` (302)
  walk inner backward. ✓

**Cleared categories re-audited:**
- **tbmv / tbsv / tpmv / tpsv** (12 files): scripted audit showed
  every descending-outer block paired with `for (int i = ...; i > j;
  --i)` (descending inner). No ascending inner under descending
  outer. ✓
- **trsm / trmm**: spot-read qtrsm.f and qtrmm.f — descending-outer
  blocks themselves walk inner ascending in the Fortran reference
  (e.g. `DO 50 K = M,1,-1` / `DO 40 I = 1,K-1` ascending). Overlay
  matches. ✓
- **symm / hemm**: false-alarm initially on xsymm.c:89-97 (descending
  outer + ascending inner). Read Fortran xsymm.f:305-321: `DO 90
  I = M,1,-1 / DO 80 K = I+1,M` — ASCENDING inner in Fortran too.
  Same pattern in esymm.f:305-321. Overlay matches Fortran. ✓
- **Rank-1 / rank-2 / gemv / sbmv / hbmv / hbmv / etc.**: grep over
  `src/parallel_blas/**` confirmed no `for (... = N-1; ... >= 0; --)`
  outer loops in any of these files. Pattern can't apply. ✓

**No false positives found. No missed sites found.** The doc accurately
captures every descending-outer-ascending-inner site in the overlay
where the Fortran reference disagrees.

## Already fixed (commit `ac6fe030`)

| File | Path | Lines |
|---|---|---|
| `src/parallel_blas/kind10/etrsv.c` | incx=1 LT | 68-91 |
| `src/parallel_blas/kind10/ytrsv.c` | incx=1 LT/LC (both conj_a branches) | 61-86 |

## Sub-class A — trsv LT/LC general-stride fallback (overlooked in `ac6fe030`)

The incx=1 paths got the fix, but the analogous incx≠1 fallbacks were
missed. Same bug, lower call frequency.

| File | Lines (outer / inner) | Path |
|---|---|---|
| `src/parallel_blas/kind10/etrsv.c` | 127 / 129 | incx≠1 LT |
| `src/parallel_blas/kind10/ytrsv.c` | 118 / 120-124 | incx≠1 LT/LC (both branches) |

**Perf impact:** real on kind10 when callers happen to pass non-unit
stride; cold in default benchmark configurations. Mechanical fix.

## Sub-class B — trsv LT/LC across other targets

Same shape as the kind10 etrsv/ytrsv fix, propagated to the remaining
two targets.

| File | Lines (outer / inner) | Path |
|---|---|---|
| `src/parallel_blas/kind16/qtrsv.c` | 60 / 63 | LTN incx=1 |
| `src/parallel_blas/kind16/qtrsv.c` | 101 / 103 | LTN incx≠1 |
| `src/parallel_blas/kind16/xtrsv.c` | 59 / 63, 66 | LTN/LCN incx=1 (conj + noconj) |
| `src/parallel_blas/kind16/xtrsv.c` | 111 / 113 | LTN/LCN incx≠1 |
| `src/parallel_blas/multifloats/mtrsv.cpp` | 153 / 157-180 | LT SIMD body |
| `src/parallel_blas/multifloats/mtrsv.cpp` | 238 / 241 | LT scalar fallback |
| `src/parallel_blas/multifloats/mtrsv.cpp` | 280 / 282 | LT incx≠1 |
| `src/parallel_blas/multifloats/wtrsv.cpp` | 214 / 177-183 | LT/LC SIMD body (via `do_dot_range`) |
| `src/parallel_blas/multifloats/wtrsv.cpp` | 248 / 252, 255 | LT/LC scalar |
| `src/parallel_blas/multifloats/wtrsv.cpp` | 301 / 303 | LT/LC incx≠1 |

**Perf impact:** none measurable today. Both kind16 (libquadmath) and
multifloats (DD software) are compute-bound — every multiply costs
~100ns / ~6-10 cycles respectively, dwarfing memory effects. If either
gets faster (hardware quad, vectorized DD), the bug surfaces.

## Sub-class C — trmv NL path (lower-tri, no-trans), writes `x[i]`

Outer `j = N-1..0` descends, inner `i = j+1..N-1` ascends and **writes**
`x[i]`. Write traffic goes through write-combining buffers so the
cache-direction sensitivity is weaker than the read case, but Fortran
still walks the inner backward at every analog site.

| File | Lines (outer / inner) | Stride |
|---|---|---|
| `src/parallel_blas/kind10/etrmv.c` | 46 / 50 | incx=1 |
| `src/parallel_blas/kind10/etrmv.c` | 106 / 108-112 | incx≠1 |
| `src/parallel_blas/kind10/ytrmv.c` | 41 / 47 | incx=1 |
| `src/parallel_blas/kind10/ytrmv.c` | 91 / 96 | incx≠1 |
| `src/parallel_blas/kind16/qtrmv.c` | 40 / 44 | incx=1 |
| `src/parallel_blas/kind16/qtrmv.c` | 81 / 84 | incx≠1 |
| `src/parallel_blas/kind16/xtrmv.c` | 38 / 42 | incx=1 |
| `src/parallel_blas/kind16/xtrmv.c` | 88 / 91 | incx≠1 |
| `src/parallel_blas/multifloats/mtrmv.cpp` | 41 / 45 | incx=1 |
| `src/parallel_blas/multifloats/mtrmv.cpp` | 82 / 85 | incx≠1 |
| `src/parallel_blas/multifloats/wtrmv.cpp` | 49 / 53 | incx=1 |
| `src/parallel_blas/multifloats/wtrmv.cpp` | 99 / 102 | incx≠1 |

**Perf impact:** uncertain — measure before/after if fixing kind10.

## Sub-class D — trmv UT/UC path (upper-tri, trans/conj-trans), reads `x[i]`

Outer `j = N-1..0` descends, inner `i = 0..j-1` ascends and **reads**
`x[i]`. Read-side, so same severity as the original trsv LT bug if
out-of-cache.

| File | Lines (outer / inner) | Stride / notes |
|---|---|---|
| `src/parallel_blas/kind10/etrmv.c` | 85 / 91-96 | incx=1 — **has 2-chain unroll** |
| `src/parallel_blas/kind10/etrmv.c` | 131 / 134-137 | incx≠1 |
| `src/parallel_blas/kind10/ytrmv.c` | 74 / 78-84 | incx=1 (conj + noconj) |
| `src/parallel_blas/kind10/ytrmv.c` | 120 / 124-127 | incx≠1 |
| `src/parallel_blas/kind16/qtrmv.c` | 68 / 72 | incx=1 |
| `src/parallel_blas/kind16/qtrmv.c` | 104 / 107 | incx≠1 |
| `src/parallel_blas/kind16/xtrmv.c` | 71 / 76, 78 | incx=1 (conj + noconj) |
| `src/parallel_blas/kind16/xtrmv.c` | 115 / 118 | incx≠1 |
| `src/parallel_blas/multifloats/mtrmv.cpp` | 69 / 73 | incx=1 |
| `src/parallel_blas/multifloats/mtrmv.cpp` | 105 / 108 | incx≠1 |
| `src/parallel_blas/multifloats/wtrmv.cpp` | 82 / 87, 89 | incx=1 (conj + noconj) |
| `src/parallel_blas/multifloats/wtrmv.cpp` | 126 / 129 | incx≠1 |

**Perf impact:** likely real for kind10 (etrmv/ytrmv) at out-of-cache N
— same mechanism as the etrsv LTN bug. `etrmv.c:85` already carries a
2-chain x87 dot product unroll; flipping its direction needs to keep
that structure (descend in pairs).

## Cleared — no action needed

Surveyed exhaustively across all three targets:

- **tbsv / tbmv / tpsv / tpmv** variants (banded and packed triangular)
  — every descending-outer block already pairs with a descending inner.
- **trsm / trmm** level-3 variants — Fortran reference itself uses
  ascending inner; overlay matches.
- **symm / hemm** diagonal-block kernels — descending outer paths fire
  only when M ≤ nb (≤64), so working set is bounded; matches Fortran.
- **gbmv / hbmv / sbmv / spmv / hpmv / symv / hemv / gemv** and all
  rank-1 / rank-2 updates (`*spr`, `*spr2`, `*syr`, `*syr2`, `*her`,
  `*her2`, `*hpr`, `*hpr2`) — outer ascends, doesn't match the pattern.

## Recommendations

1. **Sub-class A** (2 files, kind10): finish what `ac6fe030` started.
   Real perf when callers use non-unit stride.
2. **Sub-class D** for kind10 only (`etrmv.c`, `ytrmv.c`, ~4 paths):
   most likely to move kind10 numbers at large N. Preserve the existing
   2-chain unroll in `etrmv.c:85`.
3. **Sub-class C** for kind10: measure first — write-side cache
   sensitivity may not be as strong.
4. **Sub-classes B/C/D for kind16 + multifloats**: low priority. The
   pattern is wrong but the perf is dominated by software-emulated
   arithmetic, so fixes won't move benchmarks today. Worth doing as a
   single sweep if/when the rule is enforced consistently.

Workflow per fix:
- Edit source under `src/parallel_blas/...`
- Copy to corresponding `/tmp/stage-{e,q,m}/parallel_blas/...`
- Rebuild `perf_<name>` and `fuzz_<name>`
- Run `fuzz_<name>` (must pass 200/200 with zero error)
- Run `perf_<name>` at N=512,1024,2048 with iters=50, verify the
  affected key (LT/LC for trsv; NL or UT/UC for trmv) hits ≥1.0×
- Commit with the now-standard message format

## Empirical results (2026-05-18, applied)

After extending `fuzz_trsv/trmv` to exercise incx ∈ {1, 2, 3, -1, -2}
and `perf_trsv/trmv` to cover `BLAS_PERF_INCX=1,2`, each fix was
benched before/after on a kernel-isolated harness (OMP=1,
taskset -c 0).

### kind10 — applied

| Routine | Change | Sub-class | Bench result |
|---|---|---|---|
| `etrsv.c` incx≠1 LT | inner → backward | A | (only incx≠1 path; not in default bench) |
| `ytrsv.c` incx≠1 LT/LC | inner → backward | A | LTN/x2 N=512: **0.296× → 1.002×**; LCN/x2 N=512: **0.301× → 0.978×** |
| `etrmv.c` UT/UC, both strides | inner → backward, 2-chain preserved | D | within noise (UTN already 1.10-1.20× at all measured N) |
| `etrmv.c` NL, both strides | inner → backward | C | within noise |
| `ytrmv.c` UT/UC, both strides | inner → backward | D | UTN N=2048: **0.892× → 1.009×** |
| `ytrmv.c` NL, both strides | inner → backward | C | within noise |

### kind16 + multifloats — REGRESSION, not applied

The doc predicted "no measurable" impact for these targets because
both are compute-bound (libquadmath ~100 ns/mul, software DD ~6-10
cycles/mul). The measurement is actually the opposite of predicted:
**backward inner is consistently slower** for incx≠1 paths.

Direct comparison on `qtrsv` LTN/x2 (incx=2):

| N | Forward (baseline) | Backward (Rule 21) |
|---|---|---|
| 512  | 1.106× | 1.002× |
| 1024 | 1.245× | 0.975× |

`xtrsv` LTN/LCN: 1.07-1.17× forward vs 0.80-1.00× backward.

`mtrsv` LTN/x2 absolute GFLOPS (3-run mean, overlay):

| N | Forward (baseline) | Backward (Rule 21) |
|---|---|---|
| 512  | 1.71 GFs | 0.87 GFs |
| 1024 | 1.55 GFs | 0.97 GFs |
| 2048 | 1.32 GFs | 1.04 GFs |

**Mechanism (hypothesis):** the cache-direction cliff in kind10 trsv
LT happens because the FP work is fast relative to memory; x falls
out of L1 between outer iters. For kind16 / multifloats the FP work
*is* the bottleneck (libquadmath call, DD reciprocal+renorm chain),
so x stays hot regardless. Meanwhile Intel's hardware streaming
prefetcher detects ascending strides more aggressively than
descending, and the ascending walk also lets gcc pipeline the
column read against the stale x[k] read. Forward inner wins.

**Action:** kind16 (qtrsv, xtrsv, qtrmv, xtrmv) and multifloats
(mtrsv, wtrsv, mtrmv, wtrmv) keep their forward inner walks. This
diverges from Rule 21 deliberately. Annotate the relevant sites if
this is ever revisited — the divergence is empirically justified,
not an oversight.

### Rule 21 refinement

The original Rule 21 in `parallel-blas-optimization-findings-
20260515.md` reads:

> Inner-loop direction is part of the algorithm, not the style.

Refined to:

> Match the Fortran reference's inner-loop direction **for
> compute-light targets where memory bandwidth catches up to compute
> (kind10/x87)**. For compute-bound targets (kind16/libquadmath,
> multifloats DD) the cache-direction cliff does not materialize and
> ascending inner wins via the hardware prefetcher — keep forward
> regardless of the reference.

### Infrastructure changes

- `tests/blas_parallel/common/fuzz_util_body.fypp`: added
  `rand_incx()` returning {1, 2, 3, -1, -2}.
- `tests/blas_parallel/consistency/fuzz_trsv_{real,complex}_body.fypp`
  and `fuzz_trmv_{real,complex}_body.fypp`: allocate X with
  stride-aware length, plumb incx through. Without this, the
  pre-existing fuzz tested only incx=1 — the survey fixes for
  Sub-class A would have shipped untested.
- `tests/blas_parallel/perf/perf_common.h`: added
  `perf_parse_int_list()`.
- `scripts/gen_perf_harnesses.py::emit_trmv_trsv`: loop over
  `BLAS_PERF_INCX` (default `{1, 2}`). incx=1 keeps existing keys;
  others get `/xN` suffix (e.g. `LTN/x2`, `LTN/x-1`). Without this
  the perf harness couldn't see the Sub-class A and B regressions.

The fuzz extension caught the would-be regression on kind16 (drop
that backward fix before commit). The perf extension is what
revealed the kind16 + multifloats results above.
