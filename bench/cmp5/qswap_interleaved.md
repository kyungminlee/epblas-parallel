# qswap — interleaved 5-way re-sweep (kind16 real swap)

Baseline `da07fb2`. Wall time ns/call, min-of-N interleaved (par/ob alternating).
Smaller = faster. Bar: `par4/ob4 ≤ ~1.05`, no `par1/ob1` regression.

## Fix

`src/epblas-parallel/kind16/qswap.c` — two deficits, one kernel:
1. **Serial unit-stride** was an `int`-indexed **3-way** unroll: amortized loop
   overhead over 3 not 4 elems and carried a separate counter (`addl $3` +
   `cmpl`) that GCC couldn't fold into the pointer walk → `par1/ob1` 1.04–1.08.
2. **OMP path** ran an un-unrolled scalar indexed swap (`x[i*incx]`) → under-threaded
   at large N → `par4/ob4` 1.10 (16384), 1.076 (65536).

Fix = extract one 4-way `qswap_kernel` (`ptrdiff_t`, `n & -4` head + scalar tail,
strided fallback) shared by the serial entry **and** the per-thread OMP slices —
the ob `swap_kernel` / kind10 `eswap_unit` family pattern. Codegen after matches
ob (`addq $64` main, `cmpq` pointer-end compare, no counter). Bit-exact (fuzz pass).

## Numbers (REPS=7, BEFORE = 3-way int / un-unrolled OMP)

| N | par1 | ob1 | par1/ob1 | par4 | ob4 | par4/ob4 before | par4/ob4 after |
|---|---|---|---|---|---|---|---|
| 64    | 44    | 45    | 0.98 | 45    | 45    | 1.00 | 1.00 |
| 128   | 72    | 75    | 0.96 | 75    | 75    | 1.01 | 1.00 |
| 256   | 130   | 136   | 0.96 | 136   | 136   | 1.01 | 1.00 |
| 512   | 247   | 258   | 0.96 | 258   | 258   | 1.02 | 1.00 |
| 1024  | 548   | 563   | 0.97 | 562   | 562   | 1.00 | 1.00 |
| 2048  | 1220  | 1217  | 1.00 | 1218  | 1216  | 1.01 | 1.00 |
| 4096  | 2487  | 2506  | 0.99 | 2489  | 2521  | 0.99 | 0.99 |
| 16384 | 12932 | 12862 | 1.01 | 8858  | 8654  | **1.10** | **1.02** |
| 65536 | 52848 | 53592 | 0.99 | 33143 | 32904 | **1.08** | **1.01** |

`par1/ob1` ∈ [0.96, 1.01] (serial deficit closed). `par4/ob4` ≤ 1.024 every cell.
Raw: `qswap_before_raw.tsv` (3-way+un-unrolled OMP), `qswap_after_raw.tsv` (this fix).
