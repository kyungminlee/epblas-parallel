# yhpr 5-way interleaved (authoritative) — fixed: threading skew (schedule chunk)

Warmed interleaved 5-way table, min-of-5 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
`A := alpha*x*xᴴ + A` (alpha real), complex Hermitian **packed** rank-1 update,
N×N. Small `alpha=1e-6`. Ways: par1 par4 ob1 ob4 mig (mig = migrated `yhpr.f`,
from the par archive). **Wall ns/call; ratios SMALLER = par faster.** key = UPLO.
Harness `/tmp/l2d/cmp5_rank1.sh` (RT=yhpr), driver `/tmp/l2d/d5_yhpr.c`.

**Verdict: par now beats the OpenBLAS overlay in every par4/ob4 cell** and threads
~3.8× on 4 cores. Serial was already at parity (`par1/ob1` ≈ 1.0 — the complex
rank-1 loop body is small enough not to spill x87 registers when outlined, so
unlike yhpr2 no `noinline` carve-out was needed). The only gap was the
`schedule(static)` triangular load-skew (par4/ob4 1.45–1.74 at N≥256, par4/par1
~1.8×). Fixed with `schedule(static, 1)`.

**Why static,1 (not static,8 like espr):** a schedule bake-off (par4 wall ns,
min-of-5, both UPLO, N=256–1024) found this complex rank-1 body does enough fp80
work per written element (~one complex mul-add ≈ 4× the real-rank-1 cost) to hide
the adjacent-column false sharing that penalizes chunk-1 in the lighter real
`espr`. So the finest balance wins: `static,1` ties or beats `static,8` here
(s1/s8 0.94–1.00). The lighter real rank-1 espr uses `static,8`; the heavier
complex rank-2 yhpr2 also keeps `static,1`.

## After (HEAD, schedule(static, 1))

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      128 |       46309       19985       45353       45339       44885 |   1.021   0.441   0.432   1.032
U      256 |      179871       55631      178693       58926      177808 |   1.007   0.944   0.309   1.012
U      512 |      709996      191774      709036      198190      707082 |   1.001   0.968   0.270   1.004
U     1024 |     3048126      792214     3052543      821828     3047900 |   0.999   0.964   0.260   1.000

L      128 |       46514       20009       45210       45189       45271 |   1.029   0.443   0.430   1.027
L      256 |      180175       55610      178376       59612      178380 |   1.010   0.933   0.309   1.010
L      512 |      710661      191829      708052      197174      708035 |   1.004   0.973   0.270   1.004
L     1024 |     3055130      792408     3047855      825168     3052222 |   1.002   0.960   0.259   1.001
```

## Before (baseline, schedule(static))

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      256 |      179019       85681      178376       59082      177461 |   1.004   1.450   0.479   1.009
U      512 |      707986      341830      707594      196705      705488 |   1.001   1.738   0.483   1.004
U     1024 |     3043377     1396436     3046709      817854     3037503 |   0.999   1.707   0.459   1.002
L      256 |      179343       86880      178424       59603      178383 |   1.005   1.458   0.484   1.005
L      512 |      708999      342118      708408      196568      708543 |   1.001   1.740   0.483   1.001
L     1024 |     3048664     1397418     3046321      826206     3043529 |   1.001   1.691   0.458   1.002
```

## Notes

- `par4/ob4` at N=128 is ~0.44 because OpenBLAS does not thread the update at that
  size (ob4 ≈ ob1); par does, so it's ~2.3× faster there.
- Only the `incx==1` fast path's two pragmas changed (`schedule(static)` →
  `schedule(static, 1)`). Strided fallback untouched. Correctness:
  `epblas_parallel_fuzz_yhpr`, 20000 cases, OMP=4.
