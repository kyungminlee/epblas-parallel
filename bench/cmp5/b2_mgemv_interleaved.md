# B2 — multifloats mgemv unit-stride Trans threading

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5).
Wall time ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_b2_mgemv_before_raw.tsv`, `cmp5_b2_mgemv_after_raw.tsv`.

## Problem — threaded SIMD NoTrans, serial SIMD Trans

mgemv has an AVX2 SoA double-double fast path (`MBLAS_SIMD_DD`) for the
unit-stride (`incx==incy==1`) case. The **NoTrans** SIMD branch was threaded
over rows; the **Trans** SIMD branch (column dot-products) was left **serial**.
ob threads its Trans, so it overtook par on the one unit-stride Trans cell:

```
            BEFORE (unit-stride Trans, key "T")
key    N      par1     par4      ob1      ob4    par4/ob4
T     512   434776   424820  1282412   343992    1.235     ← par serial, ob threaded
T    1024  1961190  1944491  5335338  1428574    1.361
T    2048  7729333  7677118 21482325  5721284    1.342
```

`par4 ≈ par1` confirms no threading. (The strided-Trans cells `T/x-1`, `T/x2`,
… already thread via the non-SIMD `#else` path and sit at par4/ob4 ≈ 1.00.)

## Fix — thread the SIMD Trans j-loop

The Trans SIMD path computes each `y[j]` as an independent dot over the shared
read-only x SoA scratch (`x_hi`/`x_lo`). Columns are disjoint outputs → a plain
`#pragma omp parallel for if(...) schedule(static)` over `j` is race-free, and
the per-`j` SIMD reduction order is untouched → consistency unaffected.

```c
#ifdef _OPENMP
        const int use_omp_t = (N >= MGEMV_OMP_MIN && blas_omp_max_threads() > 1
                               && !omp_in_parallel());
        #pragma omp parallel for if(use_omp_t) schedule(static)
#endif
        for (int j = 0; j < N; ++j) { /* SoA dot → y[j] */ }
```

## Result

```
            BEFORE            AFTER
key    N    par4/ob4    |    par4   ob4    par4/ob4   par4/par1
T     512    1.235      |  114776  342998   0.335      0.270   (~3.7x scaling)
T    1024    1.361      |  508748 1414572   0.360      0.260
T    2048    1.342      |  2163757 5745182  0.377      0.273
```

par now **crushes** ob on unit-stride Trans (~3× faster), scaling ~3.7×.
NoTrans, strided-Trans, and all serial cells unchanged. Consistency 80/80 at
OMP 1 and 4 (max-err 2.9e-32 / 1.8e-32 = DD floor; per-j reduction order intact).

## Note — NoTrans unit-stride SIMD stays serial (par wins anyway)

The NoTrans SIMD `N`-path is also serial, but par's serial SIMD is ~6× faster
than ob there (`N 512` par1 254324 vs ob4 440122 → par4/ob4 ≈ 0.053), so it is
not a breach and is left unthreaded — its outputs overlap across columns, which
would need a private-buffer fold to thread safely; not worth it when par already
dominates.
