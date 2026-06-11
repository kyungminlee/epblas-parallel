# eger — interleaved 5-way re-sweep (kind10 real rank-1 update)

Baseline `d4a4f72`. Wall time ns/call, min-of-5 interleaved (par/ob alternating).
Smaller = faster. Bar: `par4/ob4 ≤ ~1.05`, no `par1/ob1` regression.

## Fix

`src/epblas-parallel/kind10/eger.c` — `A := alpha·x·yᵀ + A`. Already threaded
over columns (rectangular, no triangular skew), so this was **not** a coverage
gap. The breach was confined to **strided x** (`incx ≠ 1`): par kept `x[ix]`
strided fp80 loads inside the threaded inner loop, while ob copies x once to a
unit-stride buffer before threading (ger_thread.c) so every thread streams x at
stride 1.

Fix = mirror ob: when `use_omp && incx != 1`, copy x into a `malloc(M)` unit
buffer once, point the inner loop at it. Bit-exact (same values, same order,
fuzz max-err 0). Serial path unchanged — its strided load isn't on a contended
path and a per-call malloc there would only add overhead. y-stride is per-column
(`y[jy0 + j·incy]`), never on the inner loop, so it never needed a buffer.

## Numbers — strided-x keys were the 18 breaches (par4/ob4, smaller=faster)

| key | N | before | after |
|---|---|---|---|
| x-1     | 128 | 1.06 | 0.98 |
| x-1     | 256 | 1.08 | 1.00 |
| x-1     | 512 | 1.11 | 1.00 |
| x2      | 128 | 1.07 | 0.99 |
| x2      | 256 | 1.10 | 1.00 |
| x2      | 512 | 1.11 | 1.00 |
| x-1/y-1 | 512 | 1.11 | 0.98 |
| x-1/y2  | 512 | 1.11 | 0.99 |
| x2/y-1  | 512 | 1.15 | 0.99 |
| x2/y2   | 512 | 1.14 | 0.98 |

(all 6 strided-x keys × N∈{128,256,512} = 18 cells, representative rows shown).

Unit-x keys (`-`, `y-1`, `y2`) were already at par4/ob4 ≈ 1.0 and stay there.
At N=1024 every key is bandwidth-bound (A=16MB, par4/par1 ≈ 0.43) so the
strided-x load was already hidden — no change there.

Serial `par1/ob1` ∈ [0.99, 1.03] every cell (unchanged). **0 FAIL** (was 18).
Raw: `eger_before_raw.tsv`, `eger_after_raw.tsv`.
