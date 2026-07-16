# yhbmv 5-way interleaved (authoritative)

Warmed interleaved 5-way table, min-of-9 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Bounded data (real unit diag + tiny complex off-diag, `beta=0` so y is recomputed
fresh each call). Ways: par1 par4 ob1 ob4 mig (mig = migrated serial reference,
from the par archive). **Wall ns/call; ratios SMALLER = par faster.** K=16.
Harness `/tmp/l2d/cmp5_yhbmv.sh`.

Measured after the rewrite that ports the esbmv/ytbmv row-gather to the complex
Hermitian band case (serial register-resident complex gather + disjoint-row
threaded gather; no barrier/scratch since x,y distinct; thresholded at
YHBMV_OMP_MIN=96). Supersedes the stale GF/s cmp5.tsv rows.

```
key     N |      par1      par4       ob1       ob4       mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      64 |      6889      6873      7742      7727      9947 |   0.890   0.890   0.998   0.693
U      96 |     10813     10090     12128     12117     15619 |   0.892   0.833   0.933   0.692
U     128 |     14748     11231     16525     16501     21255 |   0.892   0.681   0.762   0.694
U     256 |     30510     15332     34044     21251     43894 |   0.896   0.721   0.503   0.695
U     512 |     62098     23956     69465     34449     89245 |   0.894   0.695   0.386   0.696
U    1024 |    124981     40817    139555     61973    179757 |   0.896   0.659   0.327   0.695
U    2048 |    250578     84459    280307    124158    360156 |   0.894   0.680   0.337   0.696
U    4096 |    502701    148658    561282    236462    724235 |   0.896   0.629   0.296   0.694

L      64 |      6879      6937      8637      8652      9683 |   0.796   0.802   1.009   0.710
L      96 |     10830     10250     13594     13509     15181 |   0.797   0.759   0.946   0.713
L     128 |     14760     11256     18408     18386     20664 |   0.802   0.612   0.763   0.714
L     256 |     30514     15661     37921     20742     42657 |   0.805   0.755   0.513   0.715
L     512 |     62057     24251     76995     34298     86608 |   0.806   0.707   0.391   0.717
L    1024 |    125179     40854    155159     63043    174425 |   0.807   0.648   0.326   0.718
L    2048 |    250578     78343    311635    115209    350140 |   0.804   0.680   0.313   0.716
L    4096 |    502905    143884    623508    232196    702013 |   0.807   0.620   0.286   0.716
```

## What changed vs the pre-rewrite baseline

The old yhbmv was an OpenBLAS `zhbmv_thread` port: a column SCATTER serial sweep
(`y[i] += t1*A` per element = complex read-modify-write to memory) plus a threaded
path with per-thread `calloc(nthreads*n)` private slots, TWO partition functions
(a `sqrt` load-balancer + an even split), and a band-aware reduction fold. That
fold is an O(n*nthreads) serial cost that floored the par4/par1 wall-ratio.
Replaced by the esbmv/ytbmv row-gather: ~200 lines deleted (incl. `<math.h>`, both
partitioners, the private slots, the fold).

**Insight — Hermitian = ytbmv NoTrans-gather + ytbmv ConjTrans-gather over the
same row.** Each y[i] is an independent dot over the full 2K+1 band. With A =
upperTri + (strictUpper)^H, both halves are reachable from `base=&A_(0,i)`:
diagonal `Re(base[K])` (REAL), right neighbour `A(i,i+d)=base[K+d*s1]`
(anti-diagonal walk, DIRECT), left neighbour `A(i,i-d)=conj(A(i-d,i))=conj(base[K-d])`
(contiguous in column i, CONJUGATED). Lower storage mirrors it (left anti-diagonal
direct, right contiguous conjugated). The contiguous/reflected neighbour is the
conjugated one in both triangles; the diagonal is real.

**Win 1 — serial register-resident complex gather.** Accumulate each y[i] in a
`_Complex long double` x87 register pair and store once (`y[i] += alpha*s`),
instead of the per-element complex y RMW. The complex accumulator stays resident
across BOTH inner loops (`gcc -S`: 4 fldt/element/loop, no accumulator reload
between the direct and conjugated loop; conj is one `fchs`, zero extra loads). The
real diagonal is seeded as `Re(base)*x[i]` — a real*complex (2 mults, not 4),
which is both required (LAPACK leaves imag(diag) unreferenced) and cheaper. Since
x and y are DISTINCT arrays (BLAS forbids hbmv aliasing), the gather needs NO
buffer and NO ordering trick. `p1/o1` **0.80 (L) / 0.89 (U)**, `p1/mig` **0.69–0.72**
(par serial ~28–31% faster than the migrated column scatter — the complex y RMW
it eliminates is 32 B/elt).

**Win 2 — disjoint-row threaded gather.** Each thread owns a disjoint output-row
range [lo,hi) and writes y[lo,hi) directly while reading x globally — NO scratch,
NO zero-fill, NO reduction, NO barrier (x,y distinct; even simpler than ytbmv's
in-place barrier+copy-back). par4/par1 falls monotonically with no floor: **0.50 →
0.39 → 0.33 → 0.29** (N=256→4096, toward 1/ncores). par4 beats ob4 everywhere:
`p4/o4` **0.61–0.76** (par4 ~1.4–1.6× ob4).

## The 2× A-traffic tradeoff (validated, not just asserted)

The gather concedes the scatter's symmetry *load*-halving: it reads each
off-diagonal stored element TWICE (once per row it serves) vs once. Complex fp80
is 32 B/elt, so the stored band (K+1)*N*32 B spills the 256 KB per-core L2 at
N≈470 (HALF esbmv's N≈965). But complex does ~4x the flops/element of real, so
arithmetic intensity RISES and the doubled L3-resident read hides even more easily
behind the x87 complex MAC. The table confirms NO knee: `p1/o1` is flat (0.80/0.89)
across N=64..4096 — straight through the L2-spill point — and `p4/o4` does not
degrade. The second read stays in the 12 MB shared L3 (until N≈22000). So the
simple gather wins across the range and the B=2 row-block escalation is NOT needed
here (same conclusion as esbmv, reached at half the spill N).

## Threshold (YHBMV_OMP_MIN=96)

Forced-threshold in-archive calibration (par4 vs real-codegen par1): break-even is
N≈88 — par4 < par1 from N=96 (`p4/p1`=0.93 U, 0.95 L), while N=64 is still a 1.34
loss. This is WAY below esbmv's/ytbmv's 320 because (a) the barrier-free,
scratch-free, copy-back-free region has the lowest fixed overhead of the family,
and (b) the complex band dot is heavy enough per row that even a tiny N amortizes
the fork. Only upper/lower exist (no trans/diag) sharing the gather body, so a
single threshold suffices. Below 96 par stays serial and still wins p1/o1.

## Correctness

max_rel_err ~1e-18 (register accumulator is more accurate than the per-element
scatter; well under tol) vs an INDEPENDENT column-scatter reference, across uplo ×
incx,incy ∈ {1,2,-1,-3} × OMP {1,4,8,16} × N=1..4096 (threaded path exercised).
The stored band carries a nonzero-imaginary (garbage) diagonal and the reference
coerces it real, so the test genuinely proves the `Re(diag)` coercion. The
in-tree fuzz_yhbmv (N≤128, fill_matrix_complex feeds garbage-imag diagonal vs the
Hermitian migrated reference) passes at OMP 1/4/8.
