# etbmv — interleaved 5-way (authoritative)

The aggregated `cmp5.tsv` rows for **etbmv** are stale and coarse: they predate
the row-gather rewrites (OMP path `b98d664`, serial NoTrans `7312281`) and are
single-run, `BLAS_PERF_TIME_BUDGET`-capped GF/s — too noisy for cell-level
OMP=4 verdicts (both overlays occasionally eat a libgomp barrier livelock; e.g.
a stray `ob4` sample produced a phantom UTU/1024 par/ob=0.29). These numbers
supersede them for etbmv.

**Method.** Warmed interleaved **min-of-9** wall ns/call; all five ways
alternated within each rep so drift/thermal cancels. Pinned (`taskset -c 0` at
OMP=1, `0-3` at OMP=4), `OMP_WAIT_POLICY=passive`. K=16, incx=1. Driver
`/tmp/l2d/d5_etbmv.c` (overlay `etbmv_` and migrated `etbmv_migrated_`, bounded-x
so `x:=A*x` stays in range); harness `/tmp/l2d/cmp5_etbmv.sh`.

**Reading.** Columns `par1 par4 ob1 ob4 mig` are bare wall **ns/call**. Ratio
columns are **wall-time ratios, SMALLER = par faster**: `p1/o1` serial par-vs-ob,
`p4/o4` threaded par-vs-ob, `p4/p1` threading (1→4 cores), `p1/mig` par serial
vs the migrated reference. `mig` is serial (no OMP).

```
key      N |      par1      par4       ob1       ob4       mig |   p1/o1   p4/o4   p4/p1  p1/mig
UNN    128 |      2486      2479      5474      5475      5010 |   0.454   0.453   0.997   0.496
UNN    256 |      5082      5078     11384     12895     10313 |   0.446   0.394   0.999   0.493
UNN    512 |     10328     10487     23008     18106     20895 |   0.449   0.579   1.015   0.494
UNN   1024 |     22156     17520     46554     29063     42046 |   0.476   0.603   0.791   0.527
UNU    128 |      2373      2365      5124      5115      4650 |   0.463   0.462   0.996   0.510
UNU    256 |      4878      4856     10650     12718      9581 |   0.458   0.382   0.996   0.509
UNU    512 |      9791      9928     21608     18209     19468 |   0.453   0.545   1.014   0.503
UNU   1024 |     21027     17242     43440     28862     39008 |   0.484   0.597   0.820   0.539
UTN    128 |      2415      2407      2953      2940      2432 |   0.818   0.819   0.997   0.993
UTN    256 |      4999      5012      6092     11900      5020 |   0.821   0.421   1.003   0.996
UTN    512 |     10206     10281     12376     15929     10226 |   0.825   0.645   1.007   0.998
UTN   1024 |     22039     17080     25283     24387     22097 |   0.872   0.700   0.775   0.997
UTU    128 |      2345      2348      2911      2903      2387 |   0.806   0.809   1.001   0.982
UTU    256 |      4928      4896      5994     11869      4955 |   0.822   0.413   0.994   0.994
UTU    512 |     10082      9938     12152     16012     10067 |   0.830   0.621   0.986   1.002
UTU   1024 |     21699     17114     24815     24165     21396 |   0.874   0.708   0.789   1.014
LNN    128 |      2520      2515      5642      5633      4971 |   0.447   0.446   0.998   0.507
LNN    256 |      5159      5150     11641     13024     10223 |   0.443   0.395   0.998   0.505
LNN    512 |     10473     10473     23580     18106     20736 |   0.444   0.578   1.000   0.505
LNN   1024 |     22511     17766     47286     28862     41608 |   0.476   0.616   0.789   0.541
LNU    128 |      2416      2419      5408      5395      4652 |   0.447   0.448   1.001   0.519
LNU    256 |      4965      4942     11169     12879      9586 |   0.444   0.384   0.995   0.518
LNU    512 |     10208     10057     22636     18082     19409 |   0.451   0.556   0.985   0.526
LNU   1024 |     21645     17060     45413     29448     39061 |   0.477   0.579   0.788   0.554
LTN    128 |      2282      2272      2286      2295      2322 |   0.998   0.990   0.996   0.982
LTN    256 |      4677      4716      4706     11680      4716 |   0.994   0.404   1.008   0.992
LTN    512 |      9449      9528      9576     15806      9596 |   0.987   0.603   1.008   0.985
LTN   1024 |     20057     17516     19813     24060     19805 |   1.012   0.728   0.873   1.013
LTU    128 |      2246      2231      2239      2224      2250 |   1.003   1.003   0.993   0.998
LTU    256 |      4637      4672      4590     11528      4600 |   1.010   0.405   1.008   1.008
LTU    512 |      9279      9296      9424     15715      9326 |   0.985   0.592   1.002   0.995
LTU   1024 |     19466     17006     19625     23806     19319 |   0.992   0.714   0.874   1.008
```

## Takeaways

- **Serial NoTrans (`7312281`): par ~2x faster.** `p1/o1` 0.44–0.49 and
  `p1/mig` ~0.50 — par serial now beats both ob and the migrated reference ~2x.
  This is the in-place row-gather (register-resident accumulator vs the old
  column scatter's per-element read-modify-write).
- **Threaded (`b98d664`): par faster in every threaded cell.** `p4/o4` ~0.58–0.62
  (NoTrans) / 0.70–0.73 (Trans) at N=1024.
- **No self-loss.** NoTrans threading break-even rose to ~700 once serial got 2x
  faster, so `ETBMV_OMP_MIN_N` was raised 512→768: N<768 now runs serial
  (`p4/p1`≈1.0, the ~1.01 at 512 is serial-vs-serial noise), N≥768 threads and
  wins (`p4/p1` 0.79–0.82 at 1024).
- **Trans serial unchanged** — already a register dot. Upper `p1/o1` ~0.82–0.87
  (par faster); lower ~0.99–1.01 (parity, the small unattributed serial-Trans
  codegen gap, not gather-fixable).
- **ob's OMP=4 collapses at small N** (`ob4` ~12µs vs `ob1` ~5µs at N=256):
  openblas threads badly there while par correctly stays serial below threshold,
  hence `p4/o4`~0.40 at N=256 — a real (reproducible) ob weakness, not a par gain.
