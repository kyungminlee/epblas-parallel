# ygemmtr 5-way interleaved (authoritative) — already ahead of ob, no rewrite

Warmed interleaved 5-way table, min-of-7 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Square output: C is the `UPLO` triangle of an N×N complex matrix, contraction K=N,
`beta=0`, complex `alpha=0.9+0.2i`. Ways: par1 par4 ob1 ob4 mig (mig = migrated
serial reference, from the par archive). **Wall ns/call; ratios SMALLER = par
faster.** key = UPLO+TA+TB ('C' = conjugate-transpose). Harness
`/tmp/l2d/cmp5_ygemmtr.sh`, driver `/tmp/l2d/d5_ygemmtr.c`.

**Verdict: par beats the OpenBLAS overlay in EVERY par4/ob4 cell and threads
near-ideally everywhere — no work warranted.** This mirrors ygemm (see
[[project_egemm_done]]): OpenBLAS has no tuned fp80-complex GEMM, so the "ob wins"
premise is REAL-only.

```
key      N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
LNN    128 |     4492883     1152281     5794886     4016608     4669602 |   0.775   0.287   0.256   0.962
LNN    256 |    35310689     8933100    45709060    18919911    36687195 |   0.773   0.472   0.253   0.962
LNN    512 |   283564250    71530992   366581270   112138090   294308492 |   0.774   0.638   0.252   0.963
LNN   1024 |  2409508444   612611258  3179188326   878785832  2491637034 |   0.758   0.697   0.254   0.967

LCN    128 |     4065899     1050271     4114304     2364979     4062310 |   0.988   0.444   0.258   1.001
LCN    256 |    32168170     8175339    32301273    10478015    32167771 |   0.996   0.780   0.254   1.000
LCN    512 |   258578624    65257117   259444682    69213134   259399248 |   0.997   0.943   0.252   0.997
LCN   1024 |  2303578862   585916745  2304472586   594620330  2304239756 |   1.000   0.985   0.254   1.000

LNC    128 |     4683430     1198884     5922591     5891427     4569301 |   0.791   0.203   0.256   1.025
LNC    256 |    36735656     9278945    46233967    46092574    35732916 |   0.795   0.201   0.253   1.028
LNC    512 |   296804779    74742992   373515003   372836220   288139352 |   0.795   0.200   0.252   1.030
LNC   1024 |  2505067782   636849627  3211659380  3214878653  2435759764 |   0.780   0.198   0.254   1.028

UNN    128 |     4513633     1155092     5789667     3835397     4698182 |   0.780   0.301   0.256   0.961
UNN    256 |    35380393     8947071    45708484    18303173    36830082 |   0.774   0.489   0.253   0.961
UNN    512 |   283098350    70833776   366536161   111268820   294101161 |   0.772   0.637   0.250   0.963
UNN   1024 |  2399378040   608564424  3170918273   866628040  2484104112 |   0.757   0.702   0.254   0.966

UCN    128 |     4065236     1047153     4115100     2327688     4059273 |   0.988   0.450   0.258   1.001
UCN    256 |    32171300     8164706    32348870    10531618    32167194 |   0.995   0.775   0.254   1.000
UCN    512 |   257958300    64742424   259264224    69142302   258289222 |   0.995   0.936   0.251   0.999
UCN   1024 |  2295852226   584863672  2295497441   593265630  2296857776 |   1.000   0.986   0.255   1.000

UNC    128 |     4685205     1198273     5903716     5888654     4581943 |   0.794   0.203   0.256   1.023
UNC    256 |    36767576     9284429    46180720    46063436    35741842 |   0.796   0.202   0.253   1.029
UNC    512 |   296729534    75134451   372395318   371626779   288271231 |   0.797   0.202   0.253   1.029
UNC   1024 |  2494373434   635899982  3207331811  3205970575  2429128548 |   0.778   0.198   0.255   1.027
```

## Why the egemmtr MC cap does NOT transfer

ygemmtr is structurally like **ygemm**, NOT like egemmtr. It is **not blocked**: no
NC/KC/MC tiling, no packing. `ygemmtr_` is the reference algorithm — a per-column
worker `ygemmtr_col` (K-unrolled-by-2, two independent fp80 chains per output
element to mask x87 fmul latency) — threaded with a single
`omp parallel for schedule(static,1)` over the **column j-axis**, one column per
chunk. There is no MC to cap.

And it does not need one: **par4/par1 = 0.25–0.26 in every cell, including N=128**
(≈3.9× on 4 cores). At N=128 there are 128 column-chunks finely interleaved across
the team, so `schedule(static,1)` already absorbs the triangular column-length skew
(column j holds j+1 / N−j stored elements). This is the opposite of egemmtr's
blocked ic-axis, where MC=64 left only N/64 coarse blocks and the small-N team
starved (fixed by the MC cap, 576717f). The memory hypothesis "ygemmtr shares the
threaded-MC structure, the cap may transfer" was **wrong** — checked at the source.

## Where par sits vs ob

- **par4/ob4 < 1.0 in every cell.** NN 0.29–0.70, CN 0.44–0.99, NC **0.20** (ob's
  NC orientation is ~5× slower — no fp80-complex GEMM).
- **Serial par1/ob1**: ≈0.77 on NN/NC; ≈1.00 on CN. CN (op(A)=Aᴴ) is ob's native
  packing direction, so par matches rather than beats — same pattern as egemmtr's
  TN. Still parity-or-better.

## The only soft spots (both immaterial; no fix)

1. **LCN/UCN at N=1024: p4/o4 ≈ 0.985** — par still ahead, by a hair. The CN
   orientation is where ob is strongest; par stays ≥ parity at every size.
2. **NC serial: p1/mig ≈ 1.025–1.030** — par serial ~3% behind its *own* migrated
   reference on NC (stable across all sizes, so structural, not noise). Cause: NC
   (trans_a='N', trans_b='C') runs the same K-unrolled-by-2 inner loop as NN
   (`cj[i] += t0*al0[i] + t1*al1[i]`, byte-identical), but its `t0/t1` scalars come
   from `~B_(j,l)` = `b[l*ldb+j]` — a **row-strided + conjugated** B load, vs NN's
   column-contiguous `B_(l,j)`. That is K extra strided loads per column, amortized
   over the N·K inner work, hence only ~3%.
   **Not worth fixing**: it is vs the migrated ref, not vs ob — and on this very NC
   orientation par beats ob ~5× (p4/o4 = 0.20). Touching the shared t0/t1 setup
   risks the NN path that already *beats* migrated (0.96) and threads ideally, for
   zero gain against the actual competitor.

## Correctness

No source changes — characterization only. `fuzz_ygemmtr` is the in-tree guard
(passes at OMP 1/4 on the unchanged source).
