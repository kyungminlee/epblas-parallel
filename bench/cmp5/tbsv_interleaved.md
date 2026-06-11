# tbsv — threading decision (cmp5 before-baseline, min-of-3)

**Decision: DO NOT THREAD. tbsv is correctly by-design serial.**

## The decisive fact: OpenBLAS does not thread tbsv either

Across all 136 unit-stride cells (etbsv/ytbsv/qtbsv/xtbsv × U/L × N/T(/C) × N/U × N=128..1024):

| ratio | min | median | max |
|-------|-----|--------|-----|
| **ob4/ob1** (does ob thread?) | 0.943 | **0.999** | 1.084 |
| par4/par1 (does par thread?) | — | ~1.00 | — |
| par4/ob4 (OMP gap?)          | 0.93 | **~1.00** | 1.09 |
| par1/ob1 (serial parity)     | 0.946 | **1.004** | 1.145 |

ob4/ob1 ≈ 1.0 everywhere ⇒ **OpenBLAS keeps the band triangular solve serial.** There is no OMP=4 gap to close. par already matches (par4/ob4 median ~1.0).

## Why there is nothing to thread

A banded triangular solve is **O(N·K)** work (`flops = (2K+1)·N`, K=16 in the perf driver ⇒ ~33·N, *linear* in N) with a **K-deep loop-carried recurrence**. Contrast the full triangular solves:

- **tpsv / trsv**: O(N²), off-diagonal coupling O(N) per column → genuinely threadable (and threaded — see [[project_kind10_reverse_gap]]).
- **tbsv**: O(N·K), off-diagonal coupling only O(K=16) per column → linear work, K-deep dependency → **no parallel work worth the thread-spawn overhead.**

This *resolves* (rather than leaves) the tpsv/trsv-vs-tbsv asymmetry: it is justified, not a coverage gap. OpenBLAS — a heavily optimized reference — independently made the same call. Threading tbsv would only add overhead and risk regressing the clean serial parity.

## Serial soft spots (STILL OPEN)

A small cluster where par's serial trails both ob and migrated:

- etbsv **UNN** (NoTrans-Upper): par1/ob1 1.06–1.09, par1/mig ~1.15
- etbsv **UTN** (Trans-Upper): 1.05–1.06
- ytbsv **UTU**: 1.06–1.08

Disassembly verdict (`/tmp/etbsv_par.s` vs migrated `etbsv.f.o` @0x3c0): par's inner loop is the **minimal x87 sequence** — 2 loads/iter (`fldt A; fldt X; fmulp; fsubrp; fstpt`), tmp register-resident, **equal fld count to migrated**, no spill, no branchy-abs. The gap is gfortran scheduling the *outer*-loop scaffolding marginally better on a 16-iteration band (the inner kernel runs only ~K=16 iters, so outer overhead dominates). Same root cause and family as the STILL-OPEN band/Herm serial soft spots (esbmv/yhbmv/ytbmv ~1.02–1.045; yher2-LOWER ~3.5%). The NoTrans-Upper path was already investigated in a prior session (see etbsv.c:44–47 comment — "the migrated walks backward but hits the same ~0.84× floor"). **STILL OPEN — wants a fresh angle on the outer-loop scaffolding.**

Lower-triangular paths are clean (e.g. etbsv LNN par1/mig=1.006).
