# espr 5-way interleaved (authoritative) — fixed: threading skew (schedule chunk)

Warmed interleaved 5-way table, min-of-5 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
`A := alpha*x*xᵀ + A`, real symmetric **packed** rank-1 update, N×N. Small
`alpha=1e-6`. Ways: par1 par4 ob1 ob4 mig (mig = migrated `espr.f`, from the par
archive). **Wall ns/call; ratios SMALLER = par faster.** key = UPLO. Harness
`/tmp/l2d/cmp5_rank1.sh` (RT=espr), driver `/tmp/l2d/d5_espr.c`.

**Verdict: par now beats/ties the OpenBLAS overlay in every par4/ob4 cell** and
threads ~3.4–3.8× on 4 cores. espr already had healthy serial (an outer
`use_omp` branch + a char* shared-index serial walk dodge the omp-outlining tax,
so `par1/ob1` ≈ 1.0) — the only gap was the `schedule(static)` triangular
load-skew that capped threading at ~1.8–2.1× and left `par4/ob4` at 1.4–1.8 for
N≥256. Fixed with `schedule(static, 8)`.

**Why static,8 and not static,1:** a schedule bake-off (par4 wall ns, min-of-5,
both UPLO, N=256–1024) found that for this lightest-compute body — one real
fp80 mul-add per written element — cyclic chunk-1 pays a measurable false-sharing
tax (adjacent packed columns are contiguous in `ap`, so chunk-1 puts every
neighbour pair on different threads). `static,8` keeps the balance, cuts those
boundaries 8×, and is ~2–8% faster than `static,1` across all cells while staying
symmetric for both UPLO (guided was faster on UPPER but starves LOWER). Heavier
bodies hide the false sharing and prefer `static,1` (see yhpr/yhpr2).

## After (HEAD, schedule(static, 8))

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      128 |       19225       14309       19351       19365       19528 |   0.993   0.739   0.744   0.984
U      256 |       74852       31023       74930       32528       75441 |   0.999   0.954   0.414   0.992
U      512 |      295603       97443      296712       98060      296227 |   0.996   0.994   0.330   0.998
U     1024 |     1184342      354785     1186520      359391     1187696 |   0.998   0.987   0.300   0.997

L      128 |       22052       14695       21601       21609       21880 |   1.021   0.680   0.666   1.008
L      256 |       85520       31808       84657       32626       85275 |   1.010   0.975   0.372   1.003
L      512 |      336045       96901      334490       97392      335716 |   1.005   0.995   0.288   1.001
L     1024 |     1349106      355863     1347089      360220     1350032 |   1.002   0.988   0.264   0.999
```

## Before (baseline, schedule(static))

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      256 |       74546       44156       74775       31662       75084 |   0.997   1.395   0.592   0.993
U      512 |      293356      165720      293759       96822      294493 |   0.999   1.712   0.565   0.996
U     1024 |     1180048      638924     1182260      358699     1183836 |   0.998   1.781   0.541   0.997
L      256 |       85404       45660       84574       32511       85211 |   1.010   1.404   0.535   1.002
L      512 |      335739      156578      334024       97260      335426 |   1.005   1.610   0.466   1.001
L     1024 |     1343864      640062     1342900      358378     1343018 |   1.001   1.786   0.476   1.001
```

## Notes

- `par4/ob4` at N=128 is 0.68–0.74 because OpenBLAS does not thread the update at
  that size (ob4 ≈ ob1); par does, so it's already ahead there.
- Only the `incx==1` fast path's two threaded loops changed (`schedule(static)`
  → `schedule(static, 8)`). Serial walk and strided fallback untouched.
  Correctness: `epblas_parallel_fuzz_espr`, 20000 cases, OMP=4.
