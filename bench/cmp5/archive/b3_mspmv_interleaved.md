# B3 — multifloats symmetric packed matvec (mspmv): row-gather → column-axpydot for unit stride

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5).
Wall time ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_b3_mspmv_before_raw.tsv`, `cmp5_b3_mspmv_after_raw.tsv`.

## Problem — unit-stride row-gather lost to ob's axpydot

The threaded path gathered each output row `y[i]` as an independent symmetric-row
dot reconstructed from the packed triangle. Half that row sits in column `i`
(contiguous), but the other half walks the *other* triangle by **column-jumping**
— an offset that spans the entire packed array (unbounded stride, unlike the
band case where the off-row run is bounded by the band width K). That defeats the
prefetcher, so unit-stride scaled only ~2.1× and lost to ob, whose unit-stride
path threads a contiguous-column **axpydot**:

```
            BEFORE (min-of-5)
key            par4     ob4    par4/ob4   par4/par1
mspmv L 256   156042   96596    1.615      0.476 (~2.1x)
mspmv L 512   624347  365160    1.710      0.479
mspmv L 1024 2579657 1429831    1.804      0.491
mspmv U 256   155474   96651    1.609      0.469
mspmv U 512   622960  366646    1.699      0.480
mspmv U 1024 2561610 1418340    1.806      0.494
```

The breach is **unit-stride only** — ob threads only `incx==1 && incy==1`. On
strided cells ob falls back to serial while par's row-gather *already threads and
wins ~2×*, so the strided path had to be preserved untouched.

## Fix — ob-style column-partition axpydot for unit stride

Ported ob `dspmv`/`spmv_thread.c`: threads own disjoint **column** ranges and
accumulate into a private `slot[n]`. A single contiguous pass over column `j`
does **both** the scatter (`slot[i] += x[j]*aj[i]`) and the symmetric dot
(`temp2 += aj[i]*x[i]`), reading column `j` of the packed triangle contiguously.
Per-thread slots are then AXPY-reduced with `alpha` factored into the final fold.
Column widths come from the sqrt-balanced `mspmv_partition` (port of
`symv_partition`, mask=3 min_width=4) so the triangular per-column work
(`~j` upper / `~n-j` lower) is balanced instead of skewing ~2× under equal
widths. The contiguous column read is what unlocks ~3.3× scaling.

Dispatch in `mspmv_omp`: `incx==1 && incy==1` → `mspmv_axpydot`; otherwise the
existing row-gather (which keeps the strided win). Reorders the per-row sum vs
serial → within DD fuzz tol; the serial path is unchanged and bit-exact. Fuzz
80/80 at OMP 1 and 4, max-err 0.0.

## Result

```
            BEFORE          AFTER (min-of-5)
key            par4/ob4  |  par4    ob4    par4/ob4   par4/par1
mspmv L 256    1.615     |   96486   96535   1.000      0.293  (~3.4x)
mspmv L 512    1.710     |  365889  365548   1.001      0.281
mspmv L 1024   1.804     | 1424242 1430645   0.996      0.273
mspmv U 256    1.609     |   96824   97028   0.998      0.294
mspmv U 512    1.699     |  366484  361906   1.013      0.282
mspmv U 1024   1.806     | 1414145 1417544   0.998      0.274
```

Unit-stride now **ties ob** (par4/ob4 ~1.00, was ~1.6–1.8) and scales ~3.4×
(p4/m 0.13 vs serial 0.27). Strided cells unchanged — par still **beats ob ~2×**
(e.g. L/x-1 256 par4 155563 vs ob4 322447 = 0.48; ob serializes strided). Serial
parity intact (par1/ob1 ~1.02). No FLAG on any cell. The row-gather-vs-fold
tradeoff from [[project_l2_rowgather_scaling]] resolved opposite to the band
case: for the *symmetric* matvec the off-row run is unbounded, so the
contiguous-column axpydot+fold wins over the row-gather — whereas the band's
bounded width let the fold-free restricted column-scatter win (B2).
