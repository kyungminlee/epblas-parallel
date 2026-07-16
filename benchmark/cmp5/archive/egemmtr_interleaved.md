# egemmtr 5-way interleaved (authoritative) — par ahead everywhere; small-N threading fixed

Warmed interleaved 5-way table, min-of-7 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Square output: C is the `UPLO` triangle of an N×N matrix, contraction K=N,
`beta=0`, `alpha=0.9`. Ways: par1 par4 ob1 ob4 mig (mig = migrated serial
reference, from the par archive). **Wall ns/call; ratios SMALLER = par faster.**
key = UPLO+TA+TB (e.g. LTN = lower, op(A)=Aᵀ, op(B)=B). Harness
`/tmp/l2d/cmp5_egemmtr.sh`, driver `/tmp/l2d/d5_egemmtr.c`.

**Verdict: par beats the OpenBLAS overlay in EVERY cell.** The structure was
already the egemm split (`egemmtr_serial.c` owns the math + the 4-accumulator
2×2 micro-kernel; `egemmtr_parallel.c` orchestrates threads), so egemm's TN→blocked
fix (the single-accumulator fast_col path) does not exist here to begin with — all
orientations already run through the 4-chain blocked kernel. The one fix that *did*
apply is egemm's threaded MC cap (3fb99ce): the `omp for` over the ic axis
under-split the team at small/moderate N (MC=64 ⇒ only N/64 ic-blocks) and the
triangular work-skew left the light-block threads idle under `schedule(static,1)`.

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
LNN    128 |     1066910      566511     1735088     1669033     1757922 |   0.615   0.339   0.531   0.607
LNN    256 |     7993558     2987416    13727188     7727330    13752076 |   0.582   0.387   0.374   0.581
LNN    512 |    62218791    18534837   108264414    41109860   108331254 |   0.575   0.451   0.298   0.574
LNN   1024 |   494148334   149467109   926111825   279385490   926181482 |   0.534   0.535   0.302   0.534

LTN    128 |     1069414      566860     1069758     1051423     1355957 |   1.000   0.539   0.530   0.789
LTN    256 |     7955111     2975791     8776870     4104629    10845916 |   0.906   0.725   0.374   0.733
LTN    512 |    61928833    18427214    69744919    21810107    86913408 |   0.888   0.845   0.298   0.713
LTN   1024 |   491130102   148222712   662740368   179075554   777806908 |   0.741   0.828   0.302   0.631

LNT    128 |     1057836      561502     2764455     2397642     1765272 |   0.383   0.234   0.531   0.599
LNT    256 |     7964515     3001533    21736098    11595361    13878365 |   0.366   0.259   0.377   0.574
LNT    512 |    62054350    18606235   173679621    61039119   110596184 |   0.357   0.305   0.300   0.561
LNT   1024 |   492349092   149162434  1474575722   408576626   939232193 |   0.334   0.365   0.303   0.524

UNN    128 |     1064869      562933     1732775     1597498     1753705 |   0.615   0.352   0.529   0.607
UNN    256 |     7976405     2969883    13751678     7540268    13813122 |   0.580   0.394   0.372   0.577
UNN    512 |    62164942    20252097   108135402    40226162   108345608 |   0.575   0.503   0.326   0.574
UNN   1024 |   492028715   148136100   916429913   272962350   918006292 |   0.537   0.543   0.301   0.536

UTN    128 |     1067900      551891     1068559     1052118     1356122 |   0.999   0.525   0.517   0.787
UTN    256 |     7944307     2959934     8766953     4070724    10845831 |   0.906   0.727   0.373   0.732
UTN    512 |    62023885    20103703    69724675    21701988    86886854 |   0.890   0.926   0.324   0.714
UTN   1024 |   490648907   148305234   660599035   178739053   776861303 |   0.743   0.830   0.302   0.632

UNT    128 |     1058160      561894     2757955     2377371     1747012 |   0.384   0.236   0.531   0.606
UNT    256 |     7978782     2975840    21709790    10929360    13923460 |   0.368   0.272   0.373   0.573
UNT    512 |    62127754    20224649   173476277    59218612   110243942 |   0.358   0.342   0.326   0.564
UNT   1024 |   492282100   149142851  1466373632   394686618   933341792 |   0.336   0.378   0.303   0.527
```

## The fix — threaded MC cap (lever recommended by the HPC consult)

`egemmtr_parallel.c`, immediately after `egemmtr_block_sizes`, gated on the OMP
path. Cap the resolved MC so the ic loop yields ≥~3 blocks per thread:

```c
const int nthr = blas_omp_max_threads();
if (N >= EGEMMTR_OMP_MIN && nthr > 1) {
    int cap = egemmtr_round_up((N + 3 * nthr - 1) / (3 * nthr), MR);
    if (cap < 32) cap = 32;        /* keep the register kernel amortized */
    if (MC > cap) MC = cap;
}
```

- **Rows-only retiling — bit-identical output.** MC partitions output *rows*; it
  never reorders the K-reduction (each C element's full dot accumulates in its own
  4-accumulator chain regardless of MC). `fuzz_egemmtr` passes at OMP 1 and 4.
- **Local cap.** `egemmtr_block_sizes` and `egemmtr_serial` keep MC=64; the env
  override `EGEMMTR_MC` and the serial path are untouched.
- **Floor 32, not 16.** Avoids thinning the 2×2 register kernel's amortization on
  the larger sizes (HPC consult's explicit caution).
- **No-op at large N.** N=1024 → cap 86 > 64, MC unchanged; N=1024 par4 within
  0.3% of pre-fix (no regression). Only ever lowers MC.
- **No scheduler/barrier change** — `schedule(static, 1)` and the `omp single`
  Bp-pack barrier are untouched, so the documented libgomp wedge window
  ([[project_etrsm_omp4_wedge]]) is not perturbed.

## Before → after (threading, par4/par1; smaller = better)

| N    | before | after | note |
|------|--------|-------|------|
| 128  | 0.84–0.86 (≈1.16×) | **0.52–0.53 (≈1.9×)** | MC 64→32, 2→4 ic-blocks |
| 256  | 0.46–0.47 (≈2.15×) | **0.37 (≈2.7×)** | MC 64→32, 4→8 blocks |
| 512  | 0.35 (≈2.85×) | **0.30–0.33 (≈3.2×)** | MC 64→44 |
| 1024 | 0.30 (≈3.3×) | 0.30 (unchanged) | no-op, already near ceiling |

And the one cell where par was *behind* ob pre-fix — **LTN/512 omp4 p4/o4 = 1.009**
— is now **0.845**; UTN/512 0.996 → 0.926. Every par4/ob4 cell is now < 1.0.

## Notes

- TN (op(A)=Aᵀ) is par's *weakest* orientation (p1/o1 up to ~1.0 at small N, since
  ob's TN is its native packing direction) but still parity-or-better; the NT/NN
  orientations par wins by 1.7–4×.
- N=1024 ~3.3× threading is close to the practical ceiling for this triangular
  shape on 4 cores: the perfect-balance floor is 0.25, and the smooth row-work ramp
  caps even ideal balancing around 0.26–0.27. The HPC consult judged dynamic/guided
  (≤10% at N=1024 only) and 2D jc×ic collapse (breaks shared-Bp; jc axis doesn't
  exist at N≤512) not worth the risk. MC cap was the clean lever.
- ygemmtr (complex twin) still unchecked — its kernel is complex (already 2 real
  accumulator chains, like [[project_egemm_done]]'s ygemm finding) but it shares
  the same threaded-MC structure, so the MC cap may transfer there.
