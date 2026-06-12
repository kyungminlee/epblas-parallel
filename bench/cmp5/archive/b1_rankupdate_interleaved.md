# B1 — multifloats rank-1 update schedule rebalance (mspr / msyr / wher / whpr)

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5,
`taskset -c 0` / `0-3`). Wall time in ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_b1_before_raw.tsv`, `cmp5_b1_after_raw.tsv`,
`cmp5_b1_wher_strided_after_raw.tsv`.

## Problem

The four multifloats (double-double) rank-1 updates threaded their unit-stride
column fan-out with a plain `#pragma omp parallel for schedule(static)`. Column
`j` writes `j+1` (U) / `N-j` (L) packed/strided elements — a **triangular work
skew**. Plain `static` hands each thread one contiguous block of columns, so the
thread owning the heavy end does ~2× the work of the light end: scaling caps at
~2.2× and OMP=4 **loses to the ob leg** at every N≥256.

## Fix

One-line schedule change on the unit-stride loop (serial path and the
strided `else` byte-for-byte unchanged except wher, see below):

| routine | storage          | schedule    | rationale |
|---------|------------------|-------------|-----------|
| mspr    | real packed      | `static, 8` | light real write; packed columns contiguous → cyclic `static,1` would false-share. Chunk-8 balances the skew while keeping each thread's run local. |
| msyr    | real full        | `static, 1` | full storage, columns `lda` apart → no false sharing; cyclic interleave balances the skew. |
| wher    | complex Herm full| `static, 1` | complex DD work per element is heavy → chunk-1 ideal; columns `lda` apart. |
| whpr    | complex Herm pkd | `static, 1` | complex DD work dominates any packed-column false sharing → chunk-1. |

Mirrors the kind10 twins (espr `static,8`; esyr/yher/yhpr `static,1`).

## Result — unit-stride par4/ob4 (min-of-5)

```
            BEFORE              AFTER
rt   key  N   p4/p1  p4/o4  |  p4/p1  p4/o4
mspr U 256    0.484  1.505  |  0.299  0.930
mspr U 512    0.498  1.639  |  0.297  0.966
mspr U 1024   0.462  1.647  |  0.270  0.956
mspr L 1024   0.456  1.656  |  0.273  0.951
msyr U 256    0.502  1.518  |  0.276  0.893
msyr U 1024   0.450  1.642  |  0.264  0.977
msyr L 1024   0.450  1.614  |  0.267  0.979
wher U 256    0.468  1.570  |  0.267  0.914
wher U 512    0.454  1.696  |  0.267  0.979
wher L 512    0.463  1.716  |  0.267  1.008
whpr U 256    0.460  1.500  |  0.275  0.879
whpr L 512    0.456  1.665  |  0.268  0.979
```

Scaling rose from ~2.2× (`p4/p1` ~0.45–0.50) to ~3.4–3.8× (~0.26–0.32). OMP=4
par4/ob4 dropped from **1.44–1.72** (par losing 44–72 %) to **≤1.01** across
every unit-stride cell. N=128 stays serial (below the n²≥… break-even); the
`static,8`/`static,1` choice doesn't matter there. Bit-exact (fuzz 8/8,
max-err 0.0, OMP 1+4).

## wher strided index-hoist (bonus)

The clean after-sweep surfaced a *separate* pre-existing breach: wher's strided
(`incx≠1`) path ran par4/ob4 ~1.06–1.10 — but serial `par1/ob1` was also
~1.05–1.08, so it was **serial codegen**, not a threading gap (ob doesn't thread
strided either). Root cause: the strided inner loop indexed `A_(i,j)` and
`x[kx + i*incx]`, emitting a `j*lda+i` and `kx+i*incx` index-multiply per
element. Hoisted the column base `aj=&A_(0,j)` (output A is unit-stride in i) and
walk x by incremental `ix+=incx` — mirrors the ob zher strided path, bit-identical.

Strided LOWER closed cleanly (par4/ob4 1.04–1.08 → 0.98–1.01). Strided UPPER
improved but holds a **~1.04–1.06 residual at both serial and OMP=4** — a
codegen floor on the U strided complex-Herm loop (par beats the gfortran-migrated
reference ~2× on these same cells, p4/m ~0.49). Documented soft spot, consistent
with the yher LOWER / esbmv serial residuals; not a threading defect.
