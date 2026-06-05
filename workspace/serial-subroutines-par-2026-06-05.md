# kind16 routines NOT parallelized in the `epblas-parallel` overlay

Source: full-surface min-of-5 interleaved cmp5 (`reports/cmp5/cmp5.tsv`,
2026-06-05). Both precisions, 140 routines, 5 reps, 0 timeouts / 0 partials.

**Method.** A routine runs serial in the parallel overlay when 4 threads give no
speedup, i.e. `par4/par1 ≈ 1.00` across every tested shape. **38 kind16 routines
are fully serial** by this test, plus 3 O(1) scalar generators (`qrotg`,
`qrotmg`, `xrotg`) that cannot be parallelized. A further **13 are partially
parallel** (thread unit-stride, serial on strided/Trans).

All values are **bare wall time, ns/call — smaller is faster**, taken at the
largest tested N (where the OMP gap is widest).

**How to read the columns:**

- `p4/p1 ≈ 1.00` ⇒ par took no OpenMP path (ran serial).
- `par1 ≈ ob1` ⇒ par's *serial* matches the OpenBLAS reference's serial — i.e.
  no correctness or serial-speed regression.
- `p4/ob4` ⇒ the gap that opens at OMP=4 because the OpenBLAS reference threads
  the shape while the parallel overlay does not. (`>1.0` = par slower at 4
  threads; smaller = better.)

---

## L1 reductions — kind10 threads these (`acf10a2`), kind16 never got the port

| routine | N | par1 | par4 | ob1 | ob4 | p4/p1 | p4/ob4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| qasum | 65536 | 1,016,760 | 1,015,425 | 1,005,195 | 253,612 | 1.00 | 4.00 |
| qxasum | 65536 | 2,440,412 | 2,434,953 | 2,995,674 | 760,886 | 1.00 | 3.20 |
| qdot | 65536 | 2,741,861 | 2,734,321 | 2,864,992 | 728,315 | 1.00 | 3.75 |
| xdotc | 65536 | 11,275,744 | 11,265,129 | 11,160,134 | 2,858,965 | 1.00 | 3.94 |
| xdotu | 65536 | 11,161,132 | 11,161,352 | 11,200,237 | 2,834,292 | 1.00 | 3.94 |
| iqamax | 65536 | 522,247 | 527,619 | 884,655 | 227,022 | 1.01 | 2.32 |
| ixamax | 65536 | 1,783,052 | 1,776,190 | 2,341,850 | 594,831 | 1.00 | 2.99 |

## L1 RMW — intentionally serial in *both* precisions (write-bound; par serial ties ob serial)

| routine | N | par1 | par4 | ob1 | ob4 | p4/p1 | p4/ob4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| qaxpy | 65536 | 2,451,493 | 2,454,275 | 2,455,998 | 612,284 | 1.00 | 4.01 |
| xaxpy | 65536 | 8,767,978 | 8,696,384 | 8,690,174 | 2,162,232 | 0.99 | 4.02 |
| qscal | 65536 | 1,136,141 | 1,137,970 | 1,129,273 | 293,071 | 1.00 | 3.88 |
| xscal | 65536 | 5,949,598 | 6,000,719 | 5,908,018 | 1,523,776 | 1.01 | 3.94 |
| xqscal | 65536 | 2,508,026 | 2,502,943 | 2,540,398 | 646,584 | 1.00 | 3.87 |
| qcopy | 65536 | 41,848 | 39,974 | 52,039 | 18,113 | 0.96 | 2.21 |
| xcopy | 65536 | 85,242 | 79,185 | 106,975 | 36,777 | 0.93 | 2.15 |
| qswap | 65536 | 72,688 | 69,914 | 72,633 | 44,242 | 0.96 | 1.58 |
| xswap | 65536 | 147,075 | 143,829 | 147,347 | 92,727 | 0.98 | 1.55 |
| qrot | 65536 | 7,710,118 | 7,670,206 | 7,694,770 | 1,966,553 | 0.99 | 3.90 |
| xqrot | 65536 | 16,020,989 | 15,962,440 | 15,535,811 | 3,978,658 | 1.00 | 4.01 |
| qrotm | 65536 | 7,140,713 | 7,123,423 | 7,143,962 | 1,849,747 | 1.00 | 3.85 |

## L2 sym/herm + tri/band matvec — kind10 threads many of these, kind16 doesn't

| routine | key | N | par1 | par4 | ob1 | ob4 | p4/p1 | p4/ob4 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| qsymv | L | 1024 | 45,110,735 | 45,468,998 | 44,835,072 | 11,042,222 | 1.01 | 4.12 |
| qspmv | L | 1024 | 45,128,837 | 44,872,987 | 44,864,014 | 11,038,033 | 0.99 | 4.07 |
| xhemv | L | 512 | 46,987,530 | 46,852,749 | 47,007,030 | 11,885,480 | 1.00 | 3.94 |
| xhpmv | L | 512 | 47,256,351 | 47,064,489 | 47,039,313 | 11,847,653 | 1.00 | 3.97 |
| qsbmv | L | 1024 | 1,650,948 | 1,649,108 | 1,649,116 | 524,196 | 1.00 | 3.15 |
| xhbmv | L | 512 | 3,317,822 | 3,323,977 | 3,335,351 | 972,552 | 1.00 | 3.42 |
| qtbmv | LNN | 1024 | 754,703 | 749,349 | 748,358 | 233,783 | 0.99 | 3.21 |
| xtbmv | LCN | 512 | 1,539,749 | 1,547,672 | 1,537,401 | 445,373 | 1.01 | 3.48 |
| qtrmv | LNN | 1024 | 21,270,246 | 21,150,418 | 21,085,946 | 5,778,622 | 0.99 | 3.66 |
| xtrmv | LCN | 512 | 23,368,325 | 23,354,467 | 23,411,992 | 6,184,707 | 1.00 | 3.78 |
| qtpmv | LNN | 1024 | 21,168,039 | 21,168,355 | 21,520,549 | 5,646,862 | 1.00 | 3.75 |
| xtpmv | LCN | 512 | 23,395,788 | 23,454,400 | 23,353,569 | 6,231,257 | 1.00 | 3.76 |

## L2 band/packed tri-solve — serial in par, but OpenBLAS is serial here too → no gap

| routine | key | N | par1 | par4 | ob1 | ob4 | p4/p1 | p4/ob4 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| qtbsv | LNN | 1024 | 720,844 | 720,308 | 722,874 | 724,085 | 1.00 | 0.99 |
| xtbsv | LCN | 512 | 1,676,935 | 1,678,126 | 1,689,560 | 1,686,976 | 1.00 | 0.99 |
| qtpsv | LNN | 1024 | 24,467,031 | 24,560,874 | 24,366,113 | 24,376,193 | 1.00 | 1.01 |
| xtpsv | LCN | 512 | 28,858,398 | 28,677,747 | 28,478,235 | 28,415,000 | 0.99 | 1.01 |

---

## Takeaways

- **Tri-solve four** (`q/x tbsv/tpsv`): serial in par *and* ob → `p4/ob4 ≈ 1.0`,
  **no gap at all** — not a concern.
- **L1 RMW twelve**: serial by design (matches the kind10 policy — write-bound
  RMW doesn't thread well). The omp4 "gap" is just ob threading something par
  deliberately doesn't; par's serial ties/beats ob's serial.
- **Genuinely closable gaps**: the **7 L1 reductions** and the **12 L2 matvecs**.
  All show `par1 ≈ ob1` (serial parity) but `par4/ob4` 3–4×. kind10 already
  threads its twins (`easum`/`edot`/`etbmv`/`esbmv`/… via the reduction +
  band-row-gather work); porting those paths to `parallel/kind16` would close
  them. No serial regression involved — purely missing OMP coverage.

## Partially parallel (thread unit-stride, serial on strided / Trans)

Not in the fully-serial list above, but worth noting — these 13 take the OpenMP
path on unit-stride shapes (par4/ob4 ≈ parity or par-win there) and fall back to
serial on strided (incx/incy ≠ 1) or some Trans shapes:

```
qgemv  xgemv  qgbmv  xgbmv  qger  xgerc  xgeru
qtrsv  xtrsv  qsyr  qspr  xher  xhpr
```

## Cannot be parallelized (O(1) scalar generators)

`qrotg`, `qrotmg`, `xrotg` — single-element Givens/modified-Givens generators;
no parallelism is possible or meaningful.
