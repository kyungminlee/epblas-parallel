# ygbmv 5-way interleaved (authoritative)

Warmed interleaved 5-way table, min-of-9 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Bounded data (`beta=0` so y is recomputed fresh each call). Ways: par1 par4 ob1
ob4 mig (mig = migrated serial reference, from the par archive). **Wall ns/call;
ratios SMALLER = par faster.** Square M=N, KL=KU=16. Harness `/tmp/l2d/cmp5_ygbmv.sh`.

Complex twin of egbmv. Rewrote the NoTrans path from a column SCATTER (`y[i] +=
tmp*A` per element = complex fp80 read-modify-write to memory, no OMP) to the row-
gather pattern (serial register-resident complex dot + disjoint-row threaded
gather; no barrier/scratch since x,y distinct; thresholded at YGBMV_OMP_MIN=128,
shared with Trans/ConjTrans).

```
key     N |      par1      par4       ob1       ob4       mig |   p1/o1   p4/o4   p4/p1  p1/mig
N      64 |      6843      6843     10339     10342     10326 |   0.662   0.662   1.000   0.663
N     128 |     14604     11388     22067     22035     22009 |   0.662   0.517   0.780   0.664
N     256 |     30522     15638     45566     45445     45376 |   0.670   0.344   0.512   0.673
N     512 |     63278     25482     92420     39412     92329 |   0.685   0.647   0.403   0.685
N    1024 |    126923     41270    186425     70320    186131 |   0.681   0.587   0.325   0.682
N    2048 |    254460     79109    374007    139675    372773 |   0.680   0.566   0.311   0.683
N    4096 |    510901    148734    748487    266703    746152 |   0.683   0.558   0.291   0.685

T      64 |      7119      7105      7082      7074      6880 |   1.005   1.004   0.998   1.035
T     128 |     15298     11370     15218     15223     14634 |   1.005   0.747   0.743   1.045
T     256 |     31802     15652     31643     31596     30128 |   1.005   0.495   0.492   1.056
T     512 |     64748     23697     64672     30769     61368 |   1.001   0.770   0.366   1.055
T    1024 |    130974     40679    130490     52205    123463 |   1.004   0.779   0.311   1.061
T    2048 |    261453     74413    261823    113227    247652 |   0.999   0.657   0.285   1.056
T    4096 |    527479    145212    524417    204854    501131 |   1.006   0.709   0.275   1.053

C      64 |      7414      7382      7382      7367      7685 |   1.004   1.002   0.996   0.965
C     128 |     15756     11834     15724     15735     16518 |   1.002   0.752   0.751   0.954
C     256 |     32606     16602     32489     32464     34202 |   1.004   0.511   0.509   0.953
C     512 |     66604     25745     66422     32080     69957 |   1.003   0.803   0.387   0.952
C    1024 |    133473     46707    133984     55131    141312 |   0.996   0.847   0.350   0.945
C    2048 |    268405     86082    267217    110250    281539 |   1.004   0.781   0.321   0.953
C    4096 |    538173    156228    537811    214297    566756 |   1.001   0.729   0.290   0.950
```

## What changed vs the pre-rewrite baseline

The old NoTrans path was a serial column SCATTER with NO OpenMP:
`for j: tmp=alpha*x[j]; for i in band: y[i] += tmp*A(i,j)`. The `y[i] +=` is a
complex fp80 read-modify-write to memory every element, and the j-loop carries a
cross-column dependence on y[i] so it cannot be threaded — par ran serial while
OpenBLAS threaded (the same ~2× OMP=4 loss egbmv had).

**Insight — NoTrans complex general band is the simplest gather of the family.**
No triangle, no reflection, no conjugate (A*x is plain), no real-diagonal seed:
y[i] = alpha·Σ_j A(i,j)·x[j]. With A(i,j) = a[(KU+i) + j·(lda-1)], `base = a+(KU+i)`,
`s1 = lda-1` make the row a unit-j dot `base[j*s1]*x[j]` over j ∈ [max(0,i-KL),
min(N,i+KU+1)) — the lda-1 anti-diagonal walk shared by the whole band family.

**Win 1 — serial register-resident complex gather.** Accumulate each y[i] in a
`_Complex long double` x87 register pair and store once. `gcc -S`: the hot loop is
4 fldt/element (the two complex operands), the accumulator stays resident, no
reload — no spill (the complex accumulator was the one residency risk the HPC
review flagged for the twin; confirmed clean first try). `p1/o1` **0.66–0.69** (par
serial ~1.5× faster than the OpenBLAS scatter), `p1/mig` **0.66–0.69**.

**Win 2 — disjoint-row threaded gather.** Each thread owns a disjoint output-row
range [lo,hi) and writes y[lo,hi) directly while reading x globally — NO scratch,
NO reduction, NO barrier (x,y distinct). par4/par1 falls monotonically with no
floor: **0.78 → 0.51 → 0.40 → 0.33 → 0.31 → 0.29** (N=128→4096). par4 beats ob4
everywhere: `p4/o4` **0.34–0.66** (OpenBLAS barely threads complex NoTrans at
N≤256 — ob4 ≈ ob1 there — so the par win is largest in that range). The gap is
reversed.

## Trans / ConjTrans (already a gather; threshold corrected, body un-outlined)

The Trans (`A^T*x`) and ConjTrans (`A^H*x`, conjugating A) paths were already
gathers over disjoint y[j]. Two touch-ups, no algorithm change: (1) the OMP loop
was `#pragma omp parallel for if(use_omp)`, whose `if` clause outlines the loop
body even at OMP=1 — converted to the source-level `if(use_omp){#pragma…BODY}else
{BODY}` macro duplication (egbmv Addendum 16), so the serial path inlines; (2) the
threshold moved to the shared 128. Serial stays at parity with ob (`p1/o1` ≈ 1.00
for T and C); large-N threading is healthy (`p4/o4` 0.49–0.85). T runs ~5% behind
the migrated reference serially (`p1/mig` ~1.05) — the known unattributed complex-
sweep codegen gap, left as-is; the bar is parity with ob, which holds.

## Threshold (YGBMV_OMP_MIN=128, shared N/T/C)

Forced-threshold in-archive calibration (par4 vs real-codegen par1) puts all three
paths' break-even at exactly N=128: NoTrans 96→1.000, 128→0.784; Trans 96→1.000,
128→0.734; ConjTrans 96→0.994, 128→0.739. The heavy complex band dot amortizes the
barrier-free fork early, so 128 sits far below the real egbmv's 256 (cf. the
complex Hermitian yhbmv at 96). The old 64 threaded the Trans paths in the 64–96
no-benefit range; 128 thread only where it wins. A single threshold suffices.

## Correctness

max_rel_err < 1e-15 vs an INDEPENDENT column-scatter reference, across
trans ∈ {N,T,C} (C exercises the conjugation) × rectangular M≠N × varied KL/KU
(incl. KL=0 and KU=0) × incx,incy ∈ {1,2,3,-1,-2,-3} × OMP {1,4} × N up to 480
(threaded path exercised at M≥128). Out-of-band AB cells poisoned to 1e30 catch
OOB reads. Serial gather sums ascending-j (the old column order) → bit-identical
serial output. The in-tree fuzz_ygbmv passes at OMP 1/4.
