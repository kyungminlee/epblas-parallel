# qsyrk — interleaved 5-way re-sweep (kind16 real symmetric rank-k)

Baseline `7c552e6`. Wall time ns/call, min-of-N interleaved (par/ob alternating).
Smaller = faster. Bar: `par4/ob4 ≤ ~1.05`, no `par1/ob1` regression.

## Fix

`src/epblas-parallel/kind16/qsyrk_parallel.c` — **threading orchestration only**.

The per-column compute core (`qsyrk_core`, netlib unblocked column-loop) is
**unchanged** — kind16 serial stays the Netlib reference per the design rule. The
main compute fan-out was `#pragma omp parallel for schedule(static)` over columns
`j ∈ [0,N)`. Each column's work is its triangular UPLO slice, `∝ (N-j)` (L) or
`(j+1)` (U), so contiguous `static` dumps the heavy end of the triangle on one
thread → omp4 capped at ~2.2× (`par4/par1 ≈ 0.45`) while ob's blocked 2D
partition scales ~4×. On NoTrans this lost the `par4/ob4` race (1.10–1.13).

Fix = `schedule(static, 1)`: cyclic interleave of heavy/light columns across the
team → balanced per-thread share → `par4/par1 ≈ 0.27` (~3.7×). Bitwise-identical
to the serial sweep (independent columns, read-only A); fuzz max-err 0.0.

## Numbers (NoTrans was the breach; Trans already passed)

par4/ob4 (smaller = faster, bar ≤ 1.05):

| key | N | par4/ob4 **before** | par4/ob4 **after** | par4/par1 before→after |
|---|---|---|---|---|
| LN | 64  | **1.12** | 0.68 | 0.46 → 0.28 |
| LN | 128 | **1.12** | 0.67 | 0.46 → 0.27 |
| LN | 256 | **1.13** | 0.66 | 0.45 → 0.27 |
| LN | 512 | **1.12** | 0.65 | 0.46 → 0.26 |
| UN | 64  | **1.10** | 0.67 | 0.45 → 0.28 |
| UN | 128 | **1.11** | 0.67 | 0.45 → 0.27 |
| UN | 256 | **1.13** | 0.67 | 0.45 → 0.26 |
| UN | 512 | **1.12** | 0.67 | 0.45 → 0.27 |
| LT | 64  | 0.94 | 0.56 | 0.46 → 0.27 |
| LT | 128 | 0.96 | 0.56 | 0.46 → 0.26 |
| LT | 256 | 0.96 | 0.56 | 0.46 → 0.27 |
| LT | 512 | 0.94 | 0.55 | 0.46 → 0.26 |
| UT | 64  | 0.93 | 0.56 | 0.45 → 0.27 |
| UT | 128 | 0.95 | 0.56 | 0.43 → 0.26 |
| UT | 256 | 0.96 | 0.55 | 0.45 → 0.26 |
| UT | 512 | 0.94 | 0.57 | 0.45 → 0.27 |

Serial unchanged: `par1/ob1` ∈ [0.97, 1.03] every cell (netlib core untouched).
8 NoTrans FAIL cells (before) → **0 FAIL** (after). `par4/ob4` ≤ 0.68 every cell.
Raw: `qsyrk_before_raw.tsv` (schedule static), `qsyrk_after_raw.tsv` (static,1).
