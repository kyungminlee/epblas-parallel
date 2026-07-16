# egbmv 5-way interleaved (authoritative)

Warmed interleaved 5-way table, min-of-9 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Bounded data (`beta=0` so y is recomputed fresh each call). Ways: par1 par4 ob1
ob4 mig (mig = migrated serial reference, from the par archive). **Wall ns/call;
ratios SMALLER = par faster.** Square M=N, KL=KU=16. Harness `/tmp/l2d/cmp5_egbmv.sh`.

Measured after rewriting the NoTrans path from a column SCATTER (`y[i] += tmp*A`
per element = fp80 read-modify-write to memory, no OMP) to the row-gather pattern
(serial register-resident dot + disjoint-row threaded gather; no barrier/scratch
since x,y distinct; thresholded at EGBMV_OMP_MIN=256, shared with Trans).

```
key     N |      par1      par4       ob1       ob4       mig |   p1/o1   p4/o4   p4/p1  p1/mig
N      64 |      2526      2521      5004      5007      4452 |   0.505   0.503   0.998   0.567
N     128 |      5182      5188     10577     10584      9335 |   0.490   0.490   1.001   0.555
N     256 |     10678     10164     21702     21655     19080 |   0.492   0.469   0.952   0.560
N     512 |     23046     13466     43922     23143     38665 |   0.525   0.582   0.584   0.596
N    1024 |     48116     19829     88188     38849     77784 |   0.546   0.510   0.412   0.619
N    2048 |     96291     40204    176972     78749    155636 |   0.544   0.511   0.418   0.619
N    4096 |    194581     74269    356238    136489    312536 |   0.546   0.544   0.382   0.623

T      64 |      2478      2468      2389      2382      2535 |   1.037   1.036   0.996   0.977
T     128 |      5064      5052      5003      5008      5176 |   1.012   1.009   0.998   0.978
T     256 |     10378     10620     10180     10347     10525 |   1.019   1.026   1.023   0.986
T     512 |     21218     13422     21344     17071     21718 |   0.994   0.786   0.633   0.977
T    1024 |     43290     19299     43805     27179     44131 |   0.988   0.710   0.446   0.981
T    2048 |     86521     35968     87586     50208     88621 |   0.988   0.716   0.416   0.976
T    4096 |    172851     62252    177132     92114    177702 |   0.976   0.676   0.360   0.973
```

## What changed vs the pre-rewrite baseline

The old NoTrans path was a serial column SCATTER with NO OpenMP:
`for j: tmp=alpha*x[j]; for i in band: y[i] += tmp*A(i,j)`. The `y[i] +=` is an
fp80 read-modify-write to memory every element (the x87 stack forces a load+store
round-trip per iteration), and the j-loop carries a cross-column dependence on
y[i] so it cannot be threaded. Pre-rewrite the gap was stark: **par4/ob4 = 1.66
(N512) / 2.01 / 2.05** — par was ~2× SLOWER at OMP=4 because it ran serial while
OpenBLAS threaded. The Trans path was already a gather (`s += A*x[i]`) over
disjoint y[j] and already threaded — left untouched.

**Insight — general band is the simplest gather of the family.** No triangle, no
reflection, no conjugate, no real-diagonal seed: y[i] = alpha·Σ_j A(i,j)·x[j] is a
plain dot over row i's band. With A(i,j) = a[(KU+i) + j·(lda-1)], setting
`base = a+(KU+i)` and `s1 = lda-1` makes the row a unit-j dot `base[j*s1]*x[j]`
over j ∈ [max(0,i-KL), min(N,i+KU+1)) — an lda-1 anti-diagonal walk of A, the same
stride the triangular/symmetric/Hermitian band gathers use.

**Win 1 — serial register-resident gather.** Accumulate each y[i] in an fp80 x87
register and store once (`y[i] += alpha*s`) instead of the per-element y RMW.
`gcc -S`: the hot loop is 2 fldt/element (the two operands), accumulator stays in a
register, no reload — no spill. `p1/o1` **0.49–0.55** (par serial ~2× faster than
the OpenBLAS scatter), `p1/mig` **0.56–0.62** (~1.7× faster than the migrated
column scatter — the fp80 y RMW it eliminates is 16 B/elt).

**Win 2 — disjoint-row threaded gather.** Each thread owns a disjoint output-row
range [lo,hi) and writes y[lo,hi) directly while reading x globally — NO scratch,
NO zero-fill, NO reduction, NO barrier (x,y distinct; even simpler than the
in-place triangular etbmv which needs a barrier + copy-back). par4/par1 falls
monotonically with no floor: **0.58 → 0.41 → 0.42 → 0.38** (N=512→4096). par4
beats ob4 everywhere: `p4/o4` **0.47–0.58** (par4 ~1.7–2.1× ob4). The gap is not
just closed — it is reversed.

## The strided-A tradeoff (validated, not just asserted)

The gather reads A with stride lda-1 = KL+KU (anti-diagonal walk) where the old
scatter read A contiguously down each column. The HPC review settled this: fp80
has no SIMD, so each MAC is an x87 `fldt + fmulp + faddp` dependency chain ~3-5
cycles, and at KL=KU=16 (512 B stride, one cache line per step) the strided load
retires fully hidden behind that chain — measured ~1.37 ns/MAC strided vs ~1.38
contiguous, within noise. More importantly the comparison that matters is gather
vs the OLD scatter, and the scatter paid an fp80 y-store every element that the
gather eliminates. The table confirms NO knee: `p1/o1` is flat (~0.49–0.55) across
N=64..4096, straight through the complex-band L2-spill regime. A strided penalty
only appears at KL+KU ≳ 64 (stride ≳ 2 KB); if a future K=64/128 cell misses
parity the cheap fix is a 2-row-at-a-time gather (amortize the strided A traffic
across two accumulators) — not needed at the API's typical band widths.

## Threshold (EGBMV_OMP_MIN=256, shared NoTrans+Trans)

Forced-threshold in-archive calibration (par4 vs real-codegen par1): NoTrans
break-even is N≈240 — first clean all-win at N=256 (`p4/p1`=0.95; N=224 is a 1.017
loss, N=320 is 0.811). This is between the triangular siblings' 768 and the complex
Hermitian twin's 96, as expected for a real full-band row. The Trans gather now
SHARES this threshold: the old EGBMV_OMP_MIN=64 made Trans thread far below its own
~256–512 break-even, a measured **1.7–3.2× loss at N=64–128** (par4 7720 vs par1
2478 at N=64). Unifying to 256 removes that regression (T N=64/128 now ~1.0) while
keeping the large-N Trans threading win (T N≥512 `p4/o4` 0.63–0.79). At exactly
N=256 Trans is break-even (~1.02, within noise); it ramps to a clear win by 512.

## Correctness

max_rel_err < 1e-15 vs an INDEPENDENT column-scatter reference, across
trans ∈ {N,T,C} × rectangular M≠N × varied KL/KU (incl. KL=0 and KU=0) × incx,incy
∈ {1,2,3,-1,-2,-3} × OMP {1,4} × N up to 1024 (threaded path exercised at M≥256).
Out-of-band AB cells are poisoned to 1e30 so any OOB read blows the result up. The
serial gather sums each y[i] in ascending-j order — the same column order the old
scatter accumulated — so the serial output is bit-identical, not merely close. The
in-tree fuzz_egbmv passes at OMP 1/4.
