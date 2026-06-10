# B2 — multifloats band-matvec NoTrans: row-gather → restricted column-scatter

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5).
Wall time ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_b2_band_before_raw.tsv`, `cmp5_b2_band_after_raw.tsv`.
Routines: **mtbmv** (triangular band), **mgbmv** (general band).

## Problem — strided-A read floored NoTrans scaling

Both threaded NoTrans paths gathered each output row as an independent dot,
reading the matrix **anti-diagonally** (packed stride `lda-1`). That defeats the
prefetcher, so NoTrans scaled only ~2× and lost to ob, whose NoTrans threads a
contiguous column-scatter. The Trans path (which reads its column contiguously)
was unaffected and already scaled ~3.5×, making the asymmetry explicit:

```
            BEFORE (min-of-5)
key            par4/ob4   par4/par1        | Trans sibling
mtbmv LNN 512   1.123      0.477 (~2.1x)   | LTN 512  0.743  0.285 (~3.5x)
mtbmv LNU 512   1.137      0.475           |
mtbmv UNN 512   1.066      0.453           |
mtbmv UNU 512   1.092      0.464           |
mgbmv N   512   1.092      0.449           | T   512  0.769
mgbmv N  1024   1.103      0.469           |
```

`par4 ≈ 2.1×` vs the Trans path's `~3.5×` (same routine, same band) pins the
strided row read as the limiter — not threading overhead or the DD kernel.

## Fix — restricted column-scatter (contiguous read, disjoint writes, no fold)

Each thread owns a disjoint output-row range `[lo,hi)` and walks the columns
touching it, reading every column's band segment **contiguously** (stride 1 in
the row index — exactly the layout the *serial* scatter already reads) and
scattering only into its owned rows. Writes are disjoint across threads, so
there is no race **and no O(nthreads·n) AXPY-reduce fold** (the alternative ob
uses). Iterating columns ascending (upper) / descending (lower) reproduces the
serial per-row accumulation order, so the result is **bit-exact** (fuzz 80/80 at
OMP 1 and 4, max-err 0.0 — not the DD tol floor).

- **mtbmv** (in-place triangular): NoTrans uses the new `mtbmv_colscatter`; each
  owned row is seeded at its diagonal column (assignment) before any
  off-diagonal `+=` reaches it, so the `y` scratch needs no pre-zeroing. The
  Trans path keeps the row-gather (its column read is already contiguous), now
  a dedicated `mtbmv_rowgather_t`.
- **mgbmv** (general band): NoTrans `mgbmv_n_omp` rewritten the same way. `y`
  already holds post-beta values, so each owned `y[i]` just accumulates
  `alpha*x[j]*A(i,j)` in ascending j. `alpha*x[j]` is recomputed per column from
  read-only x, which **removes the shared `ax` buffer and its barrier** (the old
  threaded path malloc'd `ax[N]` and synced before the row loop).

This is the row-gather-vs-fold tradeoff from [[project_l2_rowgather_scaling]]
resolved for the DD band case: row-gather avoids the fold but pays the strided
read; restricted column-scatter keeps the fold-free disjoint writes *and* gets
the contiguous read, because the band lets each thread bound its column sweep to
the columns intersecting its owned rows (overlap is only the band-width at block
edges, negligible).

## Result

```
            BEFORE          AFTER (min-of-5)
key            par4/ob4  |  par4    ob4    par4/ob4   par4/par1
mtbmv LNN 512   1.123    |  12454  17505    0.711      0.301  (~3.3x)
mtbmv LNN 1024  1.113    |  23717  34569    0.686      0.285
mtbmv LNU 512   1.137    |  12297  17281    0.712      0.302
mtbmv UNN 512   1.066    |  12631  18072    0.699      0.280
mtbmv UNU 512   1.092    |  12274  17691    0.694      0.295
mtbmv UNU 1024  1.090    |  23415  34902    0.671      0.278
mgbmv N   512   1.092    |  25527  37984    0.672      0.284  (~3.5x)
mgbmv N  1024   1.103    |  50007  73329    0.682      0.289
mgbmv N   128   0.51     |   8149  21184    0.385      0.378
mgbmv N   256   0.47     |  13664  43719    0.313      0.309
```

par now **beats ob ~1.4×** on every NoTrans cell (was ~1.1× slower) and scales
~3.3–3.5× (was ~2×). Trans cells unchanged (mtbmv 0.72–0.75, mgbmv 0.77 — both
untouched). Serial parity intact (par1/ob1 0.98–1.05; the mtbmv UNU ~1.05 at
N≥512 is the pre-existing serial codegen residual, serial path not modified).
Strided x/y variants (`/x-1`, `/x2`) track the unit-stride cells. Bit-exact.
