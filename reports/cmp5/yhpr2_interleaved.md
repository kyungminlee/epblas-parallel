# yhpr2 5-way interleaved (authoritative) — fixed: serial codegen + threading skew

Warmed interleaved 5-way table, min-of-5 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
`A := alpha*x*yᴴ + conj(alpha)*y*xᴴ + A`, complex Hermitian **packed** rank-2
update, N×N. Small `alpha=1e-6+0.5e-6i` so the repeated accumulation stays
bounded. Ways: par1 par4 ob1 ob4 mig (mig = migrated Fortran reference
`yhpr2.f`, from the par archive). **Wall ns/call; ratios SMALLER = par faster.**
key = UPLO. Harness `/tmp/l2d/cmp5_yhpr2.sh`, driver `/tmp/l2d/d5_yhpr2.c`.

**Verdict: par now beats the OpenBLAS overlay in EVERY par4/ob4 cell** and threads
~3.8× on 4 cores. The same espr2 pattern (commit 9ea2bb2) ported to the complex
Hermitian case, plus a third fix unique to the complex lower helper:
1. the `schedule(static)` triangular load-skew that capped threading at ~2×
   (par4/par1 0.45–0.57, par4/ob4 1.5–1.8 at N≥256 — par4 *slower* than ob4) →
   `schedule(static, 1)` cyclic, balancing both UPLO symmetrically;
2. the inlined per-column loop losing ~4–7% to the omp-parallel region spilling
   x87 operands (par1/ob1 1.04–1.07) → carve into `noinline` col helpers;
3. **(lower only, no analog in real espr2)** the naive noinline port *regressed*
   lower-serial to 1.08–1.12 — the lower helper computed the real diagonal
   *before* the loop and indexed the original arrays by the absolute `j+1..N-1`,
   so gcc walked three separate pointers (extra `addq`/iter). Fix: diagonal
   *last* + pre-advance the off-diagonal bases so the loop is 0-based over a
   single induction variable indexing three pointers — the exact tight form gcc
   auto-picks for the upper helper's `for(i=0;i<j;…)`.

## After (HEAD)

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      128 |       66188       25371       66660       66514       72406 |   0.993   0.381   0.383   0.914
U      256 |      255792       76029      262499       86169      284598 |   0.974   0.882   0.297   0.899
U      512 |     1006804      270674     1043825      303420     1129414 |   0.965   0.892   0.269   0.891
U     1024 |     4231968     1084066     4393040     1223201     4715867 |   0.963   0.886   0.256   0.897

L      128 |       63882       25056       63886       63647       61254 |   1.000   0.394   0.392   1.043
L      256 |      245986       74073      251748       81325      242380 |   0.977   0.911   0.301   1.015
L      512 |      969231      260612     1002473      279372      963444 |   0.967   0.933   0.269   1.006
L     1024 |     4045388     1060189     4217608     1134672     4043043 |   0.959   0.934   0.262   1.001
```

## Before (baseline, pre-fix: schedule(static) + inlined col loop)

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
U      128 |       70243       39795       66923       66876       72915 |   1.050   0.595   0.567   0.963
U      256 |      273808      131827      262621       87342      284923 |   1.043   1.509   0.481   0.961
U      512 |     1084427      524401     1043286      303128     1128666 |   1.039   1.730   0.484   0.961
U     1024 |     4563747     2045755     4384023     1218194     4717079 |   1.041   1.679   0.448   0.967

L      128 |       68196       37170       63747       63736       61421 |   1.070   0.583   0.545   1.110
L      256 |      264873      124452      251679       81261      242460 |   1.052   1.532   0.470   1.092
L      512 |     1045627      498426     1001090      277503      963520 |   1.044   1.796   0.477   1.085
L     1024 |     4383785     1971350     4211311     1137817     4042489 |   1.041   1.733   0.450   1.084
```

## Notes

- `par4/ob4` at N=128 is 0.38–0.39 because OpenBLAS does not thread the update at
  that size (ob4 ≈ ob1) while par does — so par4 is ~2.6× faster there. At N≥256
  ob threads too and the ratio settles to 0.88–0.93 (par still ahead).
- `par1/mig`: UPPER 0.89–0.91 (par faster than the Fortran reference), LOWER
  1.00–1.04 (par ≈ reference; the small-N excess is fork/threshold noise — the
  serial path is bit-faithful to the reference column order).
- Only the `incx==1 && incy==1` fast path was rewritten; the strided fallback is
  unchanged. Correctness: `epblas_parallel_fuzz_yhpr2`, 20000 cases, max err
  9e-20, OMP=1 and OMP=4.
