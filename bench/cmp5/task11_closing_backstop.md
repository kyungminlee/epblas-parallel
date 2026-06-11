# Task 11 closing backstop — par4/ob4 > ~1.05 gap

Date: 2026-06-10. Baseline `da07fb2`. 5-way cmp5 interleaved min-of-5 across all
19 touched routines (REPS=5; isolated cells re-confirmed at min-of-9).
Raw: `cmp5_task11_backstop_raw.tsv`. Wall time ns/call; **par/ob smaller = faster**.

**Goal:** `par4/ob4 ≤ ~1.05` in every cell, no `par1/ob1` regression, bit-exact.

## Per-routine result

| routine | cells | genuine breach | verdict |
|---|---|---|---|
| qswap  | 9  | 0 | closed |
| qsyrk  | 16 | 0 | closed (worst 0.68) |
| eger   | 36 | 0 | closed |
| esbmv  | 72 | 0 | closed (worst 0.83) |
| mger   | 36 | 0 | closed (x2/128 1.053@min-5 → 0.996@min-9, noise) |
| wgerc  | 27 | 0 | closed |
| wgeru  | 27 | 0 | closed |
| mspr   | 24 | 0 | closed |
| msyr   | 24 | 0 | closed |
| whpr   | 18 | 0 | closed |
| mtbmv  | 96 | 0 | closed |
| mgbmv  | 72 | 0 | closed (worst 0.78) |
| mspmv  | 72 | 0 | closed |
| mswap  | 9  | 0 | closed |
| mrotmg | 1  | 0 | closed |
| mrotm  | 9  | 0 | closed (-/256 1.059@min-5 → 0.989@min-9, sub-threshold noise) |
| **whemv** | 54 | 2  | **CLOSED** (5f60504) — L-strided 1.087→1.015, noinline carve |
| **wher**  | 18 | 3  | **OPEN** — U-strided serial codegen floor |
| **mgemv** | 90 | 8  | **OPEN** — small-N strided NoTrans OMP floor |

16 of 19 routines fully closed. The three with residuals are STILL-OPEN floors
on **rare strided input shapes** that need a fresh angle, and par beats the
migrated gfortran reference handily in every one:

- **whemv** — 24 cells, all `L/*` strided, `par4/ob4` ~1.08. `par1/ob1 ≈ par4/ob4`
  (≈1.08) → this is **serial**, not a threading gap (threading scales identically,
  so the ratio persists). DD error-free-transform codegen-layout/branch floor; a
  mirror experiment made it worse and non-bit-exact (reverted). `par4/mig ≈ 0.53`
  (par 1.9× faster than the migrated reference). UPPER is par-faster (~0.94).
- **wher** — 3 cells, `U/x-1` 256/512 and `U/x2` 512, `par4/ob4` ~1.05.
  `par1/ob1 ≈ par4/ob4` → serial codegen floor. `par4/mig ≈ 0.48` (par 2× faster).
- **mgemv** — 8 cells, strided NoTrans at N=128–512, `par4/ob4` 1.05–1.11.
  Inner DD-MAC self-loop **byte-identical to ob** (`len=62/fp=28/mul=3/add=11`),
  `par1/ob1` ~0.99 (serial parity-or-faster), gap closes to ~1.00 by N≥1024, and
  par dominates the common unit-stride path (NoTrans 0.55–0.81 via AVX2 SIMD,
  Trans 0.33–0.37). Small-N OMP team-dispatch floor. See
  `b2_mgemv_nstrided_interleaved.md`.

No `par1/ob1` regression on any closed routine. All fixes bit-exact (or DD
consistency-tolerance for order-changing structural ports), verified by fuzz.
