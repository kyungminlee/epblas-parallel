# ytbmv 5-way interleaved (authoritative)

Warmed interleaved 5-way table, min-of-9 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Bounded-x driver (diag=1.0 + tiny off-diag) so x:=A·x stays in normal range.
Ways: par1 par4 ob1 ob4 mig (mig = migrated serial reference, from par archive).
**Wall ns/call; ratios SMALLER = par faster.** K=16. Harness `/tmp/l2d/cmp5_ytbmv.sh`.

Measured after the rewrite that applies BOTH etbmv wins (serial in-place gather +
row-partitioned threaded gather; thresholded at YTBMV_OMP_MIN=320). Supersedes the
stale GF/s cmp5.tsv rows.

```
key      N |      par1      par4       ob1       ob4       mig |   p1/o1   p4/o4   p4/p1  p1/mig
UNN    128 |      7098      7078     11325     11300     11474 |   0.627   0.626   0.997   0.619
UNN    256 |     14677     14620     23342     19109     23684 |   0.629   0.765   0.996   0.620
UNN    384 |     22330     16654     35368     24712     35833 |   0.631   0.674   0.746   0.623
UNN    512 |     29946     18923     47455     30649     48060 |   0.631   0.617   0.632   0.623
UNN   1024 |     60852     27587     95560     54214     96611 |   0.637   0.509   0.453   0.630
UNU   1024 |     58598     26959     90761     54666     91125 |   0.646   0.493   0.460   0.643
UTN    256 |     14680     14650     14117     16808     14742 |   1.040   0.872   0.998   0.996
UTN    384 |     22859     16676     21917     21360     22662 |   1.043   0.781   0.730   1.009
UTN    512 |     30381     18908     29566     26332     30334 |   1.028   0.718   0.622   1.002
UTN   1024 |     62111     27482     60765     44840     62077 |   1.022   0.613   0.442   1.001
UTU   1024 |     59705     27464     58447     45550     59555 |   1.022   0.603   0.460   1.003
LNN   1024 |     60849     28142     96871     53930     97854 |   0.628   0.522   0.462   0.622
LNU   1024 |     58474     27443     92243     54432     93026 |   0.634   0.504   0.469   0.629
LTN   1024 |     60406     28356     58104     46633     60531 |   1.040   0.608   0.469   0.998
LTU   1024 |     57714     26873     56224     46495     57519 |   1.027   0.578   0.466   1.003
```
(Full 32-row sweep ran across UNN/UNU/UTN/UTU/LNN/LNU/LTN/LTU × {128,256,384,512,1024};
condensed here to the N ramp for UNN/UTN + every key at N=1024.)

## What changed vs the pre-rewrite baseline

Two etbmv wins applied; the old OpenBLAS `tbmv_thread` port (per-thread
`calloc(nthreads*n)` slots + sqrt/even partition tables + band-aware fold) deleted
in favour of etbmv's disjoint-output-row gather (389 → 310 lines).

**Win 1 — serial NoTrans in-place gather (register-residency).** NoTrans serial
was a column SCATTER with an `if(x[j]!=0)` guard; now an in-place band-dot gather
(upper ascending / lower descending, no buffer). `par1/ob1` NoTrans: **1.01–1.02 →
0.62–0.65** (par serial now ~37% faster than ob AND the migrated ref). Trans/
ConjTrans serial was already a register dot — left unchanged; its `par1/ob1 ≈
1.02–1.05` is the pre-existing complex-upper codegen gap (not a spill), left as-is.

**Win 2 — row-partitioned threaded gather.** Replaced the private-slot fold (which
floored `p4/p1`) with disjoint output rows → shared scratch → one barrier →
copy-back. At N=1024:
- NoTrans par4: ~41360 → **27590 ns** (~33% faster); `p4/o4` 0.76 → **0.51** (par4 ~2× ob4).
- Trans par4: ~32140 → **27480 ns** (~14% faster); `p4/o4` 0.71 → **0.61**.

**Threshold (YTBMV_OMP_MIN=320).** All three ops share one break-even (~N=290) —
no per-op split like etbmv's real case (768/1024). The fast in-place-gather serial
pushed break-even up: against the real-codegen serial, N=256 is still a marginal
loss for unit-diag/Trans, N=320 is the first N where every shape wins (~0.85–0.89).
N=256 confirmed to run serial (p4/p1 ≈ 1.00, no fork-into-loss).

## Correctness
Bit-exact (max_rel_err = 0) vs migrated: 12000 fuzz cases at OMP=4, dense-reference
checker across uplo×trans×diag×{incx} at OMP=1 (serial gather) and OMP=4 (threaded
gather), including the N=159/160/161 and 256/320/384 threshold boundaries.
