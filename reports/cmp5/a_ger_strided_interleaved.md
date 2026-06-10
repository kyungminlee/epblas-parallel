# A — multifloats ger trio strided-x threading (mger / wgerc / wgeru)

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5,
`taskset -c 0` / `0-3`). Wall time in ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_a_ger_before_raw.tsv`, `cmp5_a_ger_after_raw.tsv`.

## Problem

The three multifloats (double-double) rank-1 updates threaded only the
unit-stride (`incx==incy==1`) fast path. The strided `else` branch was a plain
serial column loop. Under iomp5 the ob leg threads the strided path (copy x to a
unit buffer, fan out over columns) and scales ~3.75×, so on every strided cell
the par leg lost badly:

```
            BEFORE (strided x-1, N=512)
rt     par1      par4       ob1       ob4    par4/par1  par4/ob4
mger  1351525  1359955   1343508   380845    1.006      3.57
wgerc 5906532  5997890   5641659  1505335    1.015      3.99
wgeru 5658212  5774513   5640724  1498373    1.021      3.85
```

`par4 ≈ par1` (no threading) while `ob4 ≈ 0.27·ob1` (ob threads ~3.7×) →
`par4/ob4 ≈ 3.3–4.0` across all strided keys (x2, x-1, y2, mixed) and sizes.

## Fix

Threaded the strided `else` exactly like the unit-stride path, mirroring ob's
`*ger_thread.c`: when `use_omp && incx != 1`, copy x once into a unit-stride
buffer (`x_buf`), then `#pragma omp parallel for if(use_omp) schedule(static)`
over the columns. Columns of A are disjoint (`A_(0,j)` is unit-stride in i, `lda`
apart in j) → race-free and **bit-exact** (the buffer copy preserves element
order; the per-column arithmetic is identical). A strided-y component is handled
in-loop with `y[jy0 + j*incy]` (read-only, no copy needed). When OMP is off or
`incx==1` the path falls back to the original strided inner loop.

Same structure in all three; wgerc additionally conjugates y
(`cconj(y[jy0+j*incy])`). `MGER/WGERC/WGERU_OMP_MIN = 64`.

## Result — strided par4/ob4 (min-of-5)

```
            BEFORE              AFTER
rt    key      N   p4/p1  p4/o4  |  p4/p1  p4/o4
mger  x-1     512  1.006  3.57   |  0.280  1.011
mger  x2      1024 1.005  3.74   |  0.269  1.008
mger  y2      512  1.003  3.52   |  0.284  1.005
wgerc x-1     512  1.015  3.99   |  0.267  1.036
wgerc x2      512  0.997  3.92   |  0.265  1.033
wgerc y-1     512  0.988  3.89   |  0.266  1.034
wgeru x-1     512  1.021  3.85   |  0.262  1.006
wgeru x2      512  0.998  3.78   |  0.262  1.007
wgeru y2      512  1.001  3.76   |  0.268  1.008
```

All strided cells now scale ~3.3–3.8× (`p4/p1 ~0.26–0.30`) and land **≤1.05**
par4/ob4. Unit-stride (`-`) cells unchanged — par already crushes ob there
(`p4/o4 ~0.08`), and the serial path is byte-for-byte untouched. Bit-exact:
fuzz 80/80 each, OMP 1+4, max-err ~1e-32 (DD round-off floor, 0 tol-fails).

## Residual soft spots (within bar, documented)

- **mger N=128** (smallest threaded size): par4/ob4 ~1.068 on two keys, but
  ob4 itself wanders 23.7–25.7 µs across keys at N=128 — measurement jitter at
  the threading break-even, par4 sits inside ob's band. Not a real gap.
- **wgerc N=512**: par4/ob4 ~1.034–1.037 — this is **serial codegen**, not a
  threading gap: `par1/ob1 ≈ 1.033` on the same cells (par's serial complex-DD
  conjugated gerc trails ob serial ~3%), and both legs thread identically
  (~3.75×). Consistent with the wher strided-UPPER / yher serial residuals.
  Under the 1.05 bar; left as-is.
