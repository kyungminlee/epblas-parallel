# B2 (b) — mgemv strided NoTrans: drop the serial precompute prologue

Date: 2026-06-10. 5-way cmp5 interleaved min-of-N (`run_gap5.sh`).
Wall time ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_b2_mgemv_nstrided_before_raw.tsv` (min-of-5),
`cmp5_b2_mgemv_nstrided_after_raw.tsv` (min-of-9).

The B2 mgemv commit threaded the SIMD **Trans** dot-loop. This closes the
separately-tracked plan sub-issue **"(b) strided N under-scaled"** — the
threaded strided **NoTrans** path (`mgemv_n_omp`).

## Diagnosis

The strided NoTrans path is the rare branch (`incx!=1 || incy!=1`, TRANS='N').
par's `mgemv_n_omp` precomputed `ax[j]=alpha*x[j]` into a `malloc`'d buffer via a
**serial prologue ahead of the parallel region**, then threaded a row-band column
scatter. ob's `gemv_n_strided` instead computes `t=alpha*x[jx]` inline inside each
thread (redundant across threads but fully overlapped) with an incremental
`iy += incy`. The fixed serial prologue (malloc + N DD-mults) does not shrink with
threads, capping par's strided scaling at ~2.8× vs ob's ~3.4× and leaving
`par4/ob4` ~1.05–1.15 at N≤512 (serial was already fine: `par1/ob1` ~0.97).

## Fix

Mirror ob `gemv_n_strided` exactly: drop the `ax[]` malloc + serial prologue,
compute `t = alpha*x[jx]` inline per thread, advance the y index incrementally.
Bit-exact (same `alpha*x` value, same ascending-j/ascending-i association;
fuzz 80/80 OMP 1 and 4, max-err unchanged DD floor ~2.6e-32).

```
                 BEFORE (min-5)        AFTER (min-9)
key          N   par4/ob4              par4/ob4
N/y-1        256   1.154                 1.021
N/x-1/y-1    256   1.133                 0.986
N/x-1/y-1    512   1.072                 0.981
N/x-1        256   1.106                 1.038
N/y2         256   1.044                 1.045
```

Worst N≤512 strided-NoTrans cell: **1.154 → 1.11** (and the N=256 column, which
held the worst cells, dropped to ~0.99–1.04).

## Residual (disassembly-backed, STILL OPEN)

Min-of-9, 8 of 40 strided-NoTrans cells remain at `par4/ob4` 1.05–1.11, all at
**N=128–512**, closing to ~1.00 by N≥1024:

```
N/x-1 128:1.058  N/x-1/y-1 512:1.066  N/x-1/y2 256:1.079  N/x2/y-1 128:1.074
N/x2/y-1 512:1.051  N/x2/y2 128:1.114  N/y-1 128:1.050  N/y-1 512:1.084
```

These are a **small-N OMP runtime floor, not a code defect**:

- par's strided inner DD-MAC self-loop is **byte-for-byte identical** to ob's:
  both `len=62, fp=28, mul=3, add=11, movsd=3` (disassembled `-O3 -march=native`).
  The fp=36 loops in ob are the *unit-stride* j-unrolled-by-2 variants, not this path.
- `par1/ob1` ~0.99 — par's serial is parity-or-faster; ob simply gains marginally
  more from threading (`par4/par1`~0.29 vs `ob4/ob1`~0.27) at these tiny absolute
  times (~28µs at N=128) where team-dispatch/barrier overhead dominates.
- The gap vanishes with N and par **dominates the common unit-stride path**:
  unit NoTrans par4/ob4 0.55–0.81 (AVX2 SIMD), unit Trans 0.33–0.37, strided Trans
  ≤1.002.

Sibling of the documented `whemv`-L / `wher`-U strided residuals
([[project_multifloats_cmp5_frontier]]): faithful structure mirror of the ob
reference, par wins serial, only a sub-2% small-N threading delta remains on a
rare input shape. STILL OPEN — needs a fresh angle on the small-N team-dispatch
floor.
