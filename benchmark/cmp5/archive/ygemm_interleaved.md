# ygemm 5-way interleaved (authoritative) — already ahead of ob, no rewrite

Warmed interleaved 5-way table, min-of-7 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Square M=N=K, `beta=0`, complex `alpha=0.9+0.2i`. Ways: par1 par4 ob1 ob4 mig
(mig = migrated serial reference, from the par archive). **Wall ns/call; ratios
SMALLER = par faster.** Harness `/tmp/l2d/cmp5_ygemm.sh`, driver `/tmp/l2d/d5_ygemm.c`.

**Verdict: ygemm beats the OpenBLAS overlay in EVERY cell — no work warranted.**
This mirrors ysymm (see [[project_l3_fused_packer_pattern]]): OpenBLAS has no tuned
fp80-complex GEMM, so the "ob wins structurally" premise is REAL-only. The par
overlay's reference-shape complex GEMM (four orientation cores, j-axis threaded)
is already 6–22% faster than ob serially and at OMP=4, and threads ~4×.

```
key     N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
NN    128 |     8880403     2247960    10204176     2628796     9247715 |   0.870   0.855   0.253   0.960
NN    256 |    70801509    17779869    80840868    20667735    73391873 |   0.876   0.860   0.251   0.965
NN    512 |   579556469   144999544   642426254   162140126   598988734 |   0.902   0.894   0.250   0.968
NN   1024 |  4846579124  1298361702  5142792666  1359700502  5016595388 |   0.942   0.955   0.268   0.966

CN    128 |     8127650     2062056    10162882     2634144     8072906 |   0.800   0.783   0.254   1.007
CN    256 |    64541252    16333366    80345976    20510281    64431846 |   0.803   0.796   0.253   1.002
CN    512 |   539950782   136452780   642046722   161895270   539377359 |   0.841   0.843   0.253   1.001
CN   1024 |  4747968095  1252324102  5136965069  1388797664  4739858522 |   0.924   0.902   0.264   1.002

NT    128 |     8916632     2273813    10184971     2649922     9548018 |   0.875   0.858   0.255   0.934
NT    256 |    70878848    17932468    80618965    20551478    76124484 |   0.879   0.873   0.253   0.931
NT    512 |   577555324   147123034   640910987   161956454   623263864 |   0.901   0.908   0.255   0.927
NT   1024 |  4782964520  1219962007  5110274196  1283847818  5147966317 |   0.936   0.950   0.255   0.929

NC    128 |     8847540     2251790    10117976     2622124     9516246 |   0.874   0.859   0.255   0.930
NC    256 |    70206598    17722527    79999782    20354946    75470780 |   0.878   0.871   0.252   0.930
NC    512 |   572288153   145273480   638420062   161153242   612874420 |   0.896   0.901   0.254   0.934
NC   1024 |  4766583235  1219541698  5102855275  1283758968  5143355456 |   0.934   0.950   0.256   0.927
```

(N/T/C combinations are symmetric in cost; NN≈NT≈NC, CN is the fastest cell. TT
not shown — same dot-core shape as CN/TN.)

## Why the egemm TN fix does NOT transfer

egemm's big win was routing the real `ta='T'` path off a single-fp80-accumulator
dot onto the 4-chain blocked kernel — a 1.74× latency-hiding gain (see
[[project_egemm_done]] and variant 4 of [[project_x87_accumulator_spill]]). That
lever is exhausted on ygemm before it starts:

- ygemm's TN/TT cores (`ygemm_tn_core`, `ygemm_tt_core`) use a SINGLE *complex*
  accumulator `acc += conj(a[l])*b[l]`. But a complex MAC already decomposes into
  TWO independent real accumulator chains (real and imaginary parts), so the
  reduction is not fadd-latency-starved the way the real single-accumulator dot
  was. `p1/mig ≈ 1.00` on CN confirms it sits exactly at the migrated reference's
  (well-scheduled) shape.
- Adding a second *complex* accumulator = 4 real chains + the complex-product
  temporaries, which overflows the 8-deep x87 stack and spills. The serial.c
  comment records this was tried and regressed; the structural reason is the
  stack depth. The real case started at 1 chain (room to grow to 4); complex
  starts at 2 (no room).

So the single-complex-accumulator dot core is already at the x87 ILP sweet spot.

## Structure (for reference)

Unlike egemm, ygemm is NOT blocked/packed: a 256² complex(10) panel is 2 MB,
larger than the L2 it would warm, and complex fp80 over-pressures the x87 stack,
so packed MR×NR tiles regressed. ygemm keeps the reference algorithm — four
orientation cores (NN/TN/NT/TT) — and parallelizes only the outer column (j) axis
(`ygemm_parallel.c`), one contiguous column slice per thread. No adaptive-MC, so
the egemm small-M single-block threading collapse cannot occur here either
(N=128/4 = 32 cols/thread). The nested-call guard (delegate to `ygemm_serial`
when `omp_in_parallel`) is the same libgomp-wedge cure as egemm.

## Correctness

No source changes were made — this is a characterization run. `fuzz_ygemm` is the
in-tree guard (passes at OMP 1/4 on the unchanged source).
