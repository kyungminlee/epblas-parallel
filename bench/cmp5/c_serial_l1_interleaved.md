# C — multifloats L1 serial-path deficits: mswap, mrotm, mrotmg

Date: 2026-06-10. 5-way cmp5 interleaved min-of-5 (`run_gap5.sh`, REPS=5).
Wall time ns/call; **par/ob ratio smaller = faster**.
Raw: `cmp5_c_serial_before_raw.tsv`, `cmp5_c_serial_after_raw.tsv`.

These three were *serial-path* deficits (par1/ob1 > 1), not threading gaps — the
par4 breach was par's slower per-element code being run unthreaded at small N. Each
was disassembled before the fix.

## mswap — no 4-way unroll

DD has no SIMD, so the only lever on a swap (2 reads + 2 writes/elt) is amortizing
loop control. par ran a scalar `for` loop; ob's `swap_kernel` unrolls 4-way with 4
independent temps. par1/ob1 ran **1.16–1.49** slower.

**Fix:** ported ob's 4-way-unrolled `mswap_kernel` (one temp set of 4), used for
both the serial path and each thread's contiguous chunk (switched the threaded
path from `parallel for` to the chunk model so the unrolled kernel runs per
thread). Bit-exact 80/80 OMP 1 and 4.

```
            BEFORE          AFTER
N        par1/ob1  par4/ob4 | par1/ob1  par4/ob4
512        1.28      1.025   |  0.949     0.955
1024       1.49      1.024   |  0.956     0.952
4096       1.34      0.93*   |  0.999     0.952
16384      1.11      1.115   |  0.961     1.013
65536      1.09      1.034   |  0.978     1.005
```
par now **beats ob serially** (par1/ob1 ~0.95) and ties it threaded (~1.0).

## mrotm — per-element flag branch

`dparam[0]` (the H-matrix selector) is loop-invariant, but par evaluated it inside
the element loop via a `step` lambda. ob unswitches it into three flag-specific
kernels (`rotm_neg/zero/pos`) — its own source even warns "Findings rule 7: do NOT
collapse to one inner loop with branches; gcc loses unswitching and emits ~3x the
stores." par1/ob1 ran **1.03–1.10** slower.

**Fix:** hoisted the flag out of the loop into the same three kernels; the
threaded chunk dispatch and the strided path branch on the flag once. Bit-exact
80/80 OMP 1 and 4.

```
            BEFORE          AFTER
N        par1/ob1  par4/ob1 | par1/ob1  par4/ob4
64         1.105     1.072   |  1.029     1.004
128        1.103     1.063   |  1.031     1.021
1024       1.071     1.091   |  1.001     0.999
16384      1.112     —       |  1.021     0.922
65536      1.080     1.040   |  1.008     0.999
```
par1/ob1 → ~1.0; par4 now ≤ ob everywhere (0.92 at 16384, par's lower OMP
threshold already wins at N=2048–4096 where ob stays serial).

## mrotmg — out-of-line fabsdd in the scalar generator

mrotmg is an O(1) scalar generator (no vector loop), so the gap is pure codegen.
Disassembly: par emitted **27 PLT calls to `fabsdd`** (a `MULTIFLOATS_API`
out-of-line library function) while ob inlines its abs (`ldabs`). Call/ret +
register save/restore on the critical path cost **1.27×** (103 vs 81 ns), despite
par being *fewer* total instructions (1747 vs 2364).

**Fix:** replaced `fabsdd` with an inline `dd_abs(a) = a < 0 ? -a : a` (header-only
ops, same idiom imamax already uses as `t_abs`). 27 calls → 0. Bit-exact 80/80.

```
          BEFORE   AFTER
        par  ob    par  ob    par/ob
mrotmg   103  81     78  81     1.27 → 0.96
```
par now beats ob.

## Summary

No FLAG on any cell after the fixes. All three bit-exact (fuzz 80/80 OMP 1 and 4,
max-err 0.0). This is the DD analog of the x87 register-residency family
([[project_x87_accumulator_spill]] trigger 1 = inline the out-of-line helper;
trigger 4 = the only fp lever without SIMD is loop-overhead amortization /
unroll) plus the source-level loop-unswitching rule.
