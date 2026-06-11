# espr2 5-way interleaved (authoritative) — fixed: serial codegen + threading skew

Warmed interleaved 5-way table, min-of-7 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
`A := alpha*x*yᵀ + alpha*y*xᵀ + A`, symmetric **packed** rank-2 update, N×N.
Small `alpha=1e-6` so the repeated accumulation stays bounded. Ways: par1 par4
ob1 ob4 mig (mig = migrated Fortran reference `espr2.f`, from the par archive).
**Wall ns/call; ratios SMALLER = par faster.** key = UPLO. Harness
`/tmp/l2d/cmp5_espr2.sh`, driver `/tmp/l2d/d5_espr2.c`.

**Verdict: par now beats the OpenBLAS overlay in EVERY par4/ob4 cell** and threads
~3–3.8× on 4 cores. Two fixes in one commit: (1) the UPPER serial ~10% codegen
deficit and (2) the `schedule(static)` triangular load-skew that capped threading
at ~2×.

## After (HEAD)

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      128 |       26177       15933       24797       24778       24799 |   1.056   0.643   0.609   1.056
U      256 |       98435       36635       96471       41160       96447 |   1.020   0.890   0.372   1.021
U      512 |      384099      119609      380227      128381      381265 |   1.010   0.932   0.311   1.007
U     1024 |     1568939      441465     1556720      470004     1561957 |   1.008   0.939   0.281   1.004

L      128 |       26480       16286       27001       27019       27470 |   0.981   0.603   0.615   0.964
L      256 |       99299       37015      106004       40119      107082 |   0.937   0.923   0.373   0.927
L      512 |      385224      123510      420543      131112      422133 |   0.916   0.942   0.321   0.913
L     1024 |     1563900      409117     1721625      480223     1734914 |   0.908   0.852   0.262   0.901
```

## Before (baseline, pre-fix)

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      256 |      107346       54925       96327       39839       96458 |   1.114   1.379   0.512   1.113
U      512 |      421596      211398      379537      127519      379755 |   1.111   1.658   0.501   1.110
U     1024 |     1711670      819004     1547986      480726     1553996 |   1.106   1.704   0.478   1.101
L      512 |      423399      212149      422917      130281      421121 |   1.001   1.628   0.501   1.005
L     1024 |     1724471      820720     1721869      466113     1722800 |   1.002   1.761   0.476   1.001
```

par4/ob4 went from **1.4–1.76 (par up to 76% slower)** to **0.85–0.94 (par faster
everywhere)**; threading par4/par1 from ~0.48–0.50 (~2×) to 0.26–0.37 (~3–3.8×).

## The two problems

### 1. Threading: `schedule(static)` triangular skew (both UPLO)

The OMP loop is over the column index `j`, but column `j` touches `j+1` (upper) or
`N−j` (lower) packed elements — a triangular work ramp. The default
`schedule(static)` hands each thread one **contiguous** block of columns, so the
heavy triangle end lands on a single thread while the rest idle. par4/par1 floored
at ~0.48 (only ~2× on 4 cores) regardless of N, while ob threaded ~3.3×.

**Fix: `schedule(static, 1)`** — cyclic distribution interleaves short and long
columns across the team, balancing the skew **symmetrically for both UPLO**. Result
~3–3.8×.

`schedule(guided)` was measured and **rejected**: it is asymmetric. Guided hands out
large contiguous chunks from `j=0` upward; that balances UPPER (heavy columns at the
high-`j` end, given out last as small chunks) but **fails LOWER** (heavy columns at
the front land in the first big chunk → par4 ≈ no threading, 744432 vs static,1's
408945 at L/1024). A work-balanced `√` partition matched static,1 but added a manual
parallel region and per-thread `sqrtl` math for no net gain — static,1 is the clean,
robust, one-keyword choice.

### 2. UPPER serial: ~10% codegen deficit from the OMP outlining

par's UPPER serial (`par1/mig ≈ 1.10`, stable across all N) was ~10% slower than its
own migrated reference running the byte-identical inner loop — **UPPER only**; LOWER
serial was already at parity. Rebuilding `espr2.c` **without** `-fopenmp` collapsed
the gap (UPPER serial 1.55e6 == mig); the `#pragma omp parallel for` outlining
degrades the UPPER inner loop's x87 register allocation (the kept-resident
`t1`/`t2`/accumulator round-trip through memory). This is variant (3) of
[[project_x87_accumulator_spill]] — inline-context register pressure.

**Fix: carve the per-column update into `noinline static` helpers**
(`espr2_col_upper` / `espr2_col_lower`). The tight loop then compiles once with
clean codegen and is shared by both the serial and threaded paths; `par1/o1` dropped
to ~1.0 (UPPER) and LOWER serial now even beats ob at large N (`par1/o1 = 0.908` at
L/1024 — par matches mig while ob's serial reference is slower there).

## Scope

Only the `incx==1 && incy==1` fast path is touched (helpers + the cyclic schedule).
The strided fallback (rare) is unchanged. Rows-only, math-preserving — `fuzz_espr2`
passes at OMP 1 and 4. Plain `omp parallel for` (no blocked_dispatch barrier), so the
libgomp wedge window ([[project_etrsm_omp4_wedge]]) is not involved.

## Addendum (2026-06-02): chunk re-tuned `static,1` → `static,8`

The "static,1 is the clean robust choice" claim above was re-examined with a
schedule bake-off (par4 wall ns, min-of-5, both UPLO, N=256–1024) across the four
packed rank-update bodies. The big win is just *balance* (plain contiguous `static`
is the only real loser); among balanced schedules the best **chunk** depends on
compute-per-written-element, because cyclic chunk-1 maximizes false sharing
(adjacent packed columns are contiguous in `ap`, so chunk-1 puts every neighbour
pair on different threads). For this **real rank-2** body `static,8` is ~1–2% faster
than `static,1` (same-session, controlled) while staying symmetric for both UPLO, so
espr2's two threaded loops were moved to `schedule(static, 8)`. The lighter real
rank-1 `espr` benefits more (~2–8%, also `static,8`); the heavier complex bodies
hide the false sharing and keep `static,1` (complex rank-1 `yhpr`, complex rank-2
`yhpr2` — where `static,8` would *regress* ~3–5%). See `reports/cmp5/espr_interleaved.md`,
`yhpr_interleaved.md`, and commit `87ca26b`. The fix-era tables above were measured
at `static,1`; espr2 remains par4/ob4 ≈ 0.89–0.91 under `static,8`.
