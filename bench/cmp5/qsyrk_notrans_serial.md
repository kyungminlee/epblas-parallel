# qsyrk NoTrans serial — NOT a deficit (cross-process measurement artifact)

Date: 2026-06-10. Task-12 Category A. Wall time ns/call; **par/ob ratio smaller =
par faster**.

## Bottom line

There is **no** qsyrk NoTrans serial deficit. The frequency-matched in-process A/B
shows **par/ob ≈ 0.94–0.96** (par 4–6% *faster* than ob) at every size and both
triangles. The "par1/ob1 ≈ 1.15" that triggered this investigation was a
**cross-process measurement artifact**, not a real gap. No code change; par's
unblocked Netlib core stays.

## How the 1.15 artifact arose

cmp5 / the perf harness measure par and ob in **separate binaries** (`perf_qsyrk`
links par + migrated; `ep_perf_qsyrk` links ob + migrated). Two process launches see
different CPU-frequency/governor states (raw ns differs ~2× run-to-run). The report
that produced 1.15 tried to cancel this by anchoring each binary to its shared
`migrated` (Netlib) leg — par/mig ÷ ob/mig — and read par/mig = 1.004, ob/mig = 0.881
⇒ 1.14. **That anchor is unreliable:** `perf_qsyrk`'s *own* par/mig was observed to
swing from 1.004 to 1.157 across builds/runs, so the cross-process division carries
that instability straight into the par/ob estimate.

## The authoritative measurement — in-process, frequency-matched

A single driver calls par's `qsyrk_serial_` and ob's `qsyrk_` on **identical data**,
in the **same process**, **alternating** per rep (min-of-20), pinned to core 2,
harness params (α=0.7, β=0.3, N=K). Because both legs run under the same frequency
state, their direct ratio is frequency-invariant:

| cell | N | par ns | ob ns | **par/ob** | maxerr(par−ob) |
|---|---|---|---|---|---|
| UN | 256 | 2.70e8 | 2.82e8 | **0.953–0.961** | 2.8e-32 |
| LN | 256 | 2.68e8 | 2.81e8 | **0.949–0.954** | 2.8e-32 |
| UN/LN | 512 | 2.14e9 | 2.28e9 | **0.937** | 6.2e-32 |

The ratio held at 0.95 across three repeats *and* across a 2× raw-frequency swing
(549 M vs 270 M ns between launches) — the invariance the cross-process harness
lacked. maxerr ~1e-32 confirms the two implementations compute the same C (≈1 ulp at
fp128), so the comparison is apples-to-apples.

Cross-check: par's parallel entry `qsyrk_` at 1 thread ≈ its `qsyrk_serial_` core
(pe/sc = 0.99–1.01), so what cmp5 exercises at omp1 is the same fast path — no
parallel-entry overhead either.

## Why ob's packed kernel does *not* win here ("if packing is bad, how is ob winning?")

It isn't winning — the premise was the artifact. And the disassembly explains why a
packed/register-tiled kernel cannot beat the plain Netlib outer-product for fp128 at
these sizes:

- **The MAC is a PLT call.** `__float128` mul/add lower to `__multf3`/`__addtf3`
  library calls (no SIMD/FMA), ~tens of cycles each, and every xmm is caller-saved —
  so accumulators **spill to stack in both implementations**. Register-tiling buys
  nothing.
- **Per-MAC memory traffic is identical.** ob's 2×2 kernel does, per MAC: 1 operand
  load + 1 acc-load + 1 acc-store + 2 calls. par's Netlib `cj[i] += t·A(i,l)` does:
  1 operand load + 1 cj-load + 1 cj-store + 2 calls. The same.
- **C-RMW is already free.** The Netlib NoTrans inner loop walks `i` (stride-1 down
  a C column that is L1-resident), so its C read-modify-write hits L1 every time. The
  C-column traffic a packed kernel "saves" wasn't costing anything.
- **Packing is pure overhead.** ob adds an O(N·K) pack/transpose pass and a blocked
  loop nest whose cache-residency win is irrelevant when compute is PLT-bound. par
  skips all of it. Net: par is ~5% faster.

This is also why the three packer experiments (per-element zero-skip, no-skip
single-accumulator, no-skip 4-way register tile) all **regressed to ~1.15** — they
bolted ob's overhead onto par's already-faster path. All reverted; `qsyrk_serial.c`
is unchanged from HEAD.

## Lesson for the harness

For routines where the cross-process subject/mig ratio is unstable, trust the
**in-process alternating A/B** (frequency-matched by construction), not the
cross-binary division. The original micro-driver that read par-faster was *correct*;
the cross-process anchor that overrode it was the artifact.
