# A — multifloats whemv strided-x serial codegen (LOWER triangle)

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5).
Wall time ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_a_whemv_before_raw.tsv`, `cmp5_a_whemv_after_raw.tsv`.

## Problem (not a threading gap)

whemv's strided path (`incx≠1` or `incy≠1`) is serial in **both** par and ob —
neither threads it (only the unit-stride SIMD path threads). So `par4≈par1`,
`ob4≈ob1`, and `par4/ob4 ≈ par1/ob1`. Profile was asymmetric by triangle:

```
            BEFORE (strided, N=256)
key        par1      ob1     par4/ob4
U/x-1   1451040  1554104    0.913   ← par already beats ob
L/x-1   1678736  1494725    1.139   ← par ~14% slower than ob
```

par **wins** the UPPER strided triangle and **loses** the LOWER one; ob has the
*opposite* asymmetry (ob's LOWER is faster than its UPPER). Unit-stride L/U
cells are healthy (par crushes ob, threads at N≥256).

## Fix — strided index hoist (bit-exact)

The strided inner loops recomputed `A_(k,i)` (= `a[i*lda+k]`) and the
`kx+k*incx` / `ky+k*incy` index-multiplies every element, and loaded A twice
(once for the temp1-axpy, once for the conj-dot). Hoisted the column base
`ai=&A_(0,i)` (A is unit-stride in the row index k → `A_(k,i)=ai[k]`), loaded
each `ai[k]` **once** into a local reused for both the axpy and `cconj(aik)`,
and replaced the per-element index-multiply with incremental `ix+=incx`,
`iy+=incy`. Applied symmetrically to both triangles. Bit-identical arithmetic
and order (fuzz 80/80, OMP 1+4, max-err 3.0e-32 = DD floor).

## Result

```
            BEFORE              AFTER
key        N    par4/ob4   |   par4/ob4
L/x-1     256    1.139     |    1.089
L/x-1     512    1.100     |    1.088
L/x2      512    1.158     |    ~1.09
U/x-1     512    0.901     |    0.937  (still beats ob)
```

LOWER strided improved ~14%→~9% (par1 ~1.68M→1.59M at N=256). UPPER and
unit-stride unchanged-or-better; serial-correct paths untouched.

## Residual — LOWER strided ~1.088 — RESOLVED (commit 5f60504)

**Update (2026-06-10):** the "codegen-layout floor" diagnosis below was WRONG.
Disassembly of the built object showed both strided inner loops spilled to
out-of-line `call cmul` + float64x2 `operator+` under register pressure inside
the 4500-insn `whemv_` body (72 calls, 33 vzeroupper), and LOWER landed the
worse basic-block layout. Carving the strided inner sweep into one `noinline`
helper (`whemv_strided_inner`, shared by both uplos) compiles it once in a clean
context with the complex-DD ops inlined. gap5 min-of-9: L-strided `par4/ob4`
avg **1.087 [1.072–1.105] → 1.015 [0.996–1.032]**, worst 1.105 → 1.032; bit-exact
(fuzz OMP1+4 relerr 0). Same fix pattern as wtbmv (`c32ca4c`). The pre-fix
analysis is retained below for the record.

## (pre-fix) Residual — LOWER strided ~1.088

After the hoist the LOWER and UPPER strided inner loops have **identical
source**, yet par's LOWER stays ~9% slower than its UPPER (and ob is opposite).
Disassembly shows the bodies are the double-double error-free-transform
sequences (two_sum / two_prod) GCC compiles with data-dependent branches
(`vucomisd` + `setnp` + `cmovne` / `je`). GCC orders the two loops' basic
blocks differently, and one layout predicts/prefetches better — a codegen-layout
/ branch floor, not a source-level spill. Confirmed: structurally mirroring
LOWER onto UPPER's shape (loop-first, combined diagonal fold) made it *worse*
(1.088→1.102) and broke bit-exactness, so it was reverted.

This sits in the documented serial-residual family (wher strided-UPPER ~1.05,
yher LOWER ~1.035, esbmv serial): par's codegen is leaner yet the migrated/ob
reference schedules the identical loop a few % better on one triangle. par
**beats the migrated reference ~2×** on these cells (p4/m ~0.53). No threading
lever applies (ob doesn't thread it either) — this is STILL OPEN and wants a
fresh angle on the serial codegen, not acceptance.
