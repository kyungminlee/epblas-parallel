# esbmv 5-way interleaved (authoritative)

Warmed interleaved 5-way table, min-of-9 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Bounded data (unit diag + tiny off-diag, `beta=0`) so repeated calls keep y bounded.
Ways: par1 par4 ob1 ob4 mig (mig = migrated serial reference, from par archive).
**Wall ns/call; ratios SMALLER = par faster.** K=16. Harness `/tmp/l2d/cmp5_esbmv.sh`.

Measured after the rewrite that ports the etbmv/ytbmv row-gather to the symmetric
band case (serial register-resident gather + disjoint-row threaded gather; no
barrier/scratch since x,y distinct; thresholded at ESBMV_OMP_MIN=320). Supersedes
the stale GF/s cmp5.tsv rows.

```
key     N |      par1      par4       ob1       ob4       mig |   p1/o1   p4/o4   p4/p1  p1/mig
U     128 |      4889      4879      6057      6054      6167 |   0.807   0.806   0.998   0.793
U     256 |     10114     10077     12488     12925     12675 |   0.810   0.780   0.996   0.798
U     320 |     12600     11145     15738     14470     15912 |   0.801   0.770   0.884   0.792
U     512 |     20560     12725     25335     19032     25663 |   0.812   0.669   0.619   0.801
U    1024 |     42938     19362     51103     30389     51732 |   0.840   0.637   0.451   0.830
U    2048 |     87912     32746    102409     57637    103262 |   0.858   0.568   0.372   0.851
U    4096 |    175459     57175    206076    100074    207338 |   0.851   0.571   0.326   0.846

L     128 |      5043      5043      6428      6434      6403 |   0.785   0.784   1.000   0.788
L     256 |     10378     10327     13239     12850     13184 |   0.784   0.804   0.995   0.787
L     320 |     13025     10679     16648     14373     16593 |   0.782   0.743   0.820   0.785
L     512 |     20996     13228     26941     18672     26670 |   0.779   0.708   0.630   0.787
L    1024 |     42580     19048     53979     30124     53494 |   0.789   0.632   0.447   0.796
L    2048 |     85926     31651    107738     53879    107354 |   0.798   0.587   0.368   0.800
L    4096 |    172322     60250    217481    104531    215873 |   0.792   0.576   0.350   0.798
```

## What changed vs the pre-rewrite baseline

The old esbmv was an OpenBLAS `dsbmv_thread` port: a column SCATTER serial sweep
(`y[i] += t1*A` per element = read-modify-write to memory) plus a threaded path
with per-thread `calloc(nthreads*n)` private slots, two partition functions (a
`sqrt` load-balancer + an even split), and a band-aware reduction fold. That fold
is an O(n*nthreads) serial cost that floored the par4/par1 wall-ratio. Replaced by
the etbmv/ytbmv row-gather: ~210 lines deleted.

**Insight — symmetric = NoTrans-gather + Trans-gather over the same row.** Each
y[i] is an independent dot over the full 2K+1 band. With only the upper triangle
stored, both halves are reachable from `base=&A_(0,i)`: diagonal `base[K]`, right
neighbour `A(i,i+d)=base[K+d*s1]` (anti-diagonal walk, s1=lda-1), left neighbour
`A(i,i-d)=A(i-d,i)=base[K-d]` (contiguous in column i). Lower storage mirrors it
(left anti-diagonal, right contiguous). Because A = upperTri + (strictUpper)^T.

**Win 1 — serial register-resident gather.** Accumulate each y[i] in an x87
register, store once (`y[i] += alpha*s`), instead of the per-element y RMW. Since
x and y are DISTINCT arrays (BLAS forbids sbmv aliasing), the gather needs NO
buffer and NO ordering trick. `p1/o1` **0.78–0.86** (par serial ~16–22% faster
than ob), `p1/mig` **0.79–0.85** (~16–21% faster than the migrated column scatter).

**Win 2 — disjoint-row threaded gather.** Each thread owns a disjoint output-row
range [lo,hi) and writes y[lo,hi) directly while reading x globally — NO scratch,
NO zero-fill, NO reduction, NO barrier (x,y distinct; even simpler than etbmv's
in-place path). par4/par1 falls monotonically with no floor: **0.62 → 0.45 → 0.37
→ 0.33** (N=512→4096, still falling toward 1/nthreads). par4 beats ob4 everywhere:
`p4/o4` **0.57–0.81** (par4 ~1.75× ob4 at N=4096).

## The 2× A-traffic tradeoff (validated, not just asserted)

The gather concedes the scatter's symmetry *load*-halving: it reads each
off-diagonal stored element TWICE (once per row it serves) vs once. The HPC review
flagged a possible L2-spill knee (A band = 272·N bytes spills the 256 KB per-core
L2 at N≈965). The N=2048/4096 probes confirm NO knee: `p1/o1` stays flat and
`p4/o4` *improves* with N. The second read stays in the 12 MB shared L3 (until
N≈44000), and on fp80 (x87, no SIMD) the L3-resident read is far cheaper than the
per-element y RMW it eliminates. So the simple gather wins across the range and the
B=2 row-block escalation (which would recover the single-load) is NOT needed here.

## Threshold (ESBMV_OMP_MIN=320)

Forced-threshold in-archive calibration (par4 vs real-codegen par1): break-even is
~N=256 (p4/p1≈1.00), first clean all-win at N=320 (~0.82–0.88). Lower than etbmv's
768 because esbmv does the full 2K+1 band per row (vs etbmv's triangular K+1), so
threading amortizes the fork sooner. Only upper/lower exist (no trans/diag) sharing
the gather body, so a single threshold suffices. In the gap [256,320) par stays
serial and still wins p4/o4: at N=256 par serial 10114 < ob 4-thread 12925.

## Correctness
max_rel_err ~1e-19 (register accumulator is ~K·ulp more accurate than the scatter,
well under the 1e-15 tol) vs the dense symmetric reference, across uplo × incx,incy
∈ {1,2,-1,-3} × OMP {1,4,8,16,64} × N=1..4096, including the 256/288/319/320/321
threshold boundary.
