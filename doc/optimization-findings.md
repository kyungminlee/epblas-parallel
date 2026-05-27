# Parallel BLAS overlay — optimization findings

Date: 2026-05-15
Branch: `parallel-blas`
Hardware: Intel i3-1315U (Raptor Lake-U, Alder Lake-derived; 2P+4E cores, P-core L1d=48 KB, L2=1.25 MB, L3=10 MB shared)

This document records what was tried, what worked, and what didn't, while tuning the GEMM overlays. It complements [`design.md`](design.md) (the design doc). Read this when you're about to "obviously" optimize a kernel — many of the obvious moves have already been tried and rejected on bench.

The unifying theme: **structural choices that paid off for one (precision, complex-ness) pair often regressed another**. Don't assume.

---

## TL;DR results table

GFLOPs at OMP=1, median speedup over migrated, range across 9 transpose combos × sizes 64/128/256(/512):

| routine | precision  | shape                  | median over migrated | shape of curve   |
|---------|------------|------------------------|----------------------|------------------|
| egemm   | kind10 real    | OpenBLAS-style packed + MR=2/NR=2 tile + TN fast-path | **1.70×** | wins everywhere ≥ s=128 |
| ygemm   | kind10 complex | reference + OMP-over-j, rank-1 paths K-unrolled by 2 | **1.00×** | parity, zero regressions |
| qgemm   | kind16 real    | reference + OMP-over-j (libquadmath-bound)            | ~1.0× | parity, libquadmath dominates |
| xgemm   | kind16 complex | reference + OMP-over-j                                 | ~1.0× | parity |
| mgemm   | mf real    | full GotoBLAS structure, MR=4 register tile via AVX2 FMA | **~28×** | hits 3.86 GF/s asymptote at s≥128 |
| wgemm   | mf complex | same as mgemm with 4-buf SoA pack (rh/rl/ih/il)        | **~19×** | hits 3.9 GF/s asymptote at s≥128 |

---

## Findings by precision

### kind10 real (egemm) — full OpenBLAS structure pays off

The original packed kernel sat at ~1.6 GFLOPs across all transposes, with migrated zgemm reference hitting 2.4 GFLOPs on `TN`/`CN` due to stride-1 access.

What worked:

1. **GotoBLAS three-level cache blocking + copy-and-conquer packing** (MC=64, KC=256, NC=512). Same shape as OpenBLAS — `js → ls → is` loop order, pack B once per (jc, pc), pack A per (ic, pc).
2. **MR=2, NR=2 register-tile outer-product micro-kernel** with 4 scalar `long double` accumulators kept on the x87 register stack across K. The compiler emits 4 independent FMA chains per K iter, masking x87 fmul latency. **This single change lifted asymptote from 1.6 → 2.2 GF/s (+40%)**.
3. **Adaptive MC when K ≤ KC**: grow `MC` so `MC*KC ≈ L2_target` (768 KB target, 4× cap). Helps small-K shapes.
4. **Explicit `TA='T', TB='N'` reference fast-path**: when A is transposed and B is not, the inner k-loop already accesses both A and B stride-1 — packing adds cost without recovering anything. The reference DDOT loop wins at TN/CN by ~5–7% over packed. Kept this fast-path despite "OpenBLAS doesn't have it" because OpenBLAS has an asm kernel that we can't match.

Result: median 1.70× over migrated across 36 (transpose, size) combos; only outlier is CN s=64 at 0.88× (single-run noise).

### kind10 complex (ygemm) — packed structure *regressed*; reverted to reference+OMP

The same MR=2 NR=2 register-tile that worked for egemm **regressed** ygemm from 1.6 → 1.5 GF/s.

**Why**: each `_Complex long double` multiply needs ~6 fp80 temp slots on the x87 stack. Four complex accumulators = 8 fp80 slots = the entire 8-deep x87 register stack, leaving no room for inputs → every input load spills to L1. The win that comes from "scalar accumulators kept in registers across K" simply doesn't exist for complex long double.

Replaced with **reference algorithm + `#pragma omp parallel for` over j**, four explicit orientation branches:

- `NN` rank-1 update (`temp = alpha*B[l,j]; c[i,j] += temp * A[i,l]`)
- `(T|C)N` dot product
- `N(T|C)` rank-1 update with B-row broadcast
- dual-transpose dot product

What further worked: **K-axis manual unroll by 2 on the rank-1 paths** (NN, NT, NC):

```c
for (l = 0; l + 1 < K; l += 2) {
    t0 = alpha * b[j2*ldb + l];
    t1 = alpha * b[j2*ldb + l + 1];
    al0 = &a[l * lda];
    al1 = &a[(l + 1) * lda];
    for (i = 0; i < M; ++i)
        cj[i] += t0 * al0[i] + t1 * al1[i];
}
```

This matches what gfortran emits for Netlib's reference zgemm rank-1 loop — two independent FMA chains per `i` mask the ~10-cycle x87 fmul latency. Recovered NN/NT/NC from **0.78× → 1.08×**.

What did *not* work and was reverted:
- **K-axis unroll on the dot-product paths** (T*N, dual-transpose): single-accumulator form was already well-scheduled by gcc; splitting to `acc0 + acc1` regressed CT/CC to 0.35×.
- **I-axis unroll on rank-1**: gcc already pipelines independent C cells across rows; manual i-unroll buys nothing.

Result: median 1.00×, range 0.97×–1.09×, **zero regressions** across all 36 combos.

### kind16 (qgemm, xgemm) — leave alone

Both already use reference + OMP-over-j. libquadmath function-call overhead per `__multf3` dominates per-op time (~hundreds of cycles), so structural optimizations (blocking, register tiling, K-unroll) **cannot help** — there's no latency to hide and no cache pressure to ease.

The K-axis unroll trick that helped ygemm rank-1 does not transfer: libquadmath calls don't pipeline within the CPU reorder window the way x87 fmuls do. Two `__multf3` calls "in flight" run sequentially.

Bench confirms: q/xgemm sit at 0.95–1.04× of migrated. No change tried; none expected to help.

### multifloats (mgemm, wgemm) — already OpenBLAS-style, one good knob

Both implement the full GotoBLAS structure from day one:
- SoA-packed Ap (column-major MR-row tile) and Bp (4 separate arrays for wgemm: rh/rl/ih/il)
- Templated MR×NR_PAN register-tile SIMD micro-kernel using `simd_dd::dd_mul` + `dd_add` (AVX2+FMA intrinsics from `mgemm_simd_kernel.h`)
- Sub-NC chunking by NR×NR_PAN
- Loop order js → ls → is
- BETA pre-pass

**Only one change shipped**: `MBLAS_SIMD_MR` default 3 → 4. Bench at 80 iters pinned to a P-core:
- s=128: +5.07% (stdev 0.38%)
- s=256: +4.65% (stdev 0.20%)
- s=64: −1.22% (stdev 3.09% — statistically indistinguishable)

MR=4 uses 8 acc + 4 broadcasts + 2 B-loads = 14 ymm of 16 available. The bigger tile amortizes per-tile setup (writeback, register fills) better at s≥128.

---

## Optimizations that did NOT work

These were tested empirically and rejected. **Don't re-try without a different motivating observation.**

### 1. Persistent thread-local scratch buffers (multifloats)
**Hypothesis**: at s=64, three `aligned_alloc` + `free` pairs per call (Ap=128 KB, Bp_hi/lo=256 KB each) dominate per-call overhead.

**Test**: replaced with `static thread_local Scratch { T*Ap; double*Bp_hi; double*Bp_lo; ... }`, sized on first call and reused.

**Result**: s=64 NN 2.56 → 2.55 GF/s. **Zero change**.

**Why**: glibc's tcache + large-bin recycling makes the 128–256 KB alloc/free pairs ~1 µs after warmup. Not the bottleneck.

### 2. OMP region disabled (multifloats)
**Hypothesis**: `GOMP_parallel` setup/teardown adds 5–20 µs/call even at `OMP_NUM_THREADS=1`.

**Test**: defined `MGEMM_DISABLE_OMP` to `#ifdef`-out all four `_OPENMP` guards in `mgemm.cpp`. Compiled, rebuilt, bench.

**Result**: NN s=64 +11% (but NT, NC, CN: 0% to −3%). Mean of 9 combos: **statistical noise**.

**Why**: libgomp's single-thread path is near-trivial. Not a real source of per-call overhead.

### 3. MR=3 tail-row hypothesis (multifloats)
**Hypothesis**: at M=64 with MR=3, the M%MR=1 tail row goes through the scalar fallback in `inner_kernel_simd_t`. Eliminating it (MR=4 divides 64 evenly) should help s=64.

**Result**: MR=4 shipped for the s≥128 gain (see multifloats section above), but the motivating s=64 case **regressed** 1.2% (within noise). The tail row was never the bottleneck; at small M the extra register pressure (8 acc + 4 broadcasts) eats scheduling slack and offsets the no-tail-row win.

### 4. simd_writeback scratch round-trip avoidance (multifloats)
**Hypothesis**: `_mm256_store_pd(ph_a, ph); ... r.limbs[0] = ph_a[j];` causes a SIMD-store→scalar-load round-trip with store-forwarding stall.

**Test**: replaced with gcc vector-element indexing `((v4df)ph)[j]` to extract lanes directly from the ymm register, no memory round-trip.

**Result**: **0% change** across all sizes (±0.3% noise).

**Why**: Alder Lake's store-forward path handles this case well. The round-trip wasn't the bottleneck — the **scalar `dd_add` on column-strided C** was. Confirmed by stubbing `simd_writeback` entirely to a no-op: that gave +6.7% at s=256, all of which is in the C read-modify-write itself, not the alpha-scale or lane extract.

### 5. Adaptive MC for multifloats
**Hypothesis**: same heuristic that helped egemm — grow MC when K is small so MC*KC ≈ L2.

**Test**: applied to mgemm with `if (K <= KC) target_mc = L2_TARGET / (K * sizeof(T))`. Bench at s=64, 128.

**Result**: s=64: noise (M=64 is already ≤ MC, so the grown MC is capped to M in the ic loop — no effect). s=128: −1 to +4%, noise.

**Why**: mgemm's MR=3 (4) micro-kernel hits asymptote at s=128 already; adaptive MC doesn't move the L2 working set into a faster regime at these sizes.

### 6. MR×NR register tile for kind10 complex (ygemm)
The egemm MR=2 NR=2 tile regressed ygemm 1.6 → 1.5 GF/s — x87 stack capacity exhausted by complex-multiply temps. Full discussion in the kind10 complex section above.

### 7. K-axis unroll on ygemm dot-product paths
K-unroll helped ygemm rank-1 (NN/NT/NC) but regressed dot-product paths. Full discussion (and the contrasting positive result on real esyrk) in the "K-axis 2-unroll" section in Addendum 1 below.

---

## Diagnostic methodology that worked

When chasing a perf gap, follow this order:

1. **Measure the actual time/cycle cost first.** Compute `cycles/flop = freq / GFLOPs`. The difference between observed and asymptote gives the absolute per-call overhead in cycles or µs. For mgemm at s=64: 2.5 vs 3.7 GF/s = 0.58 extra cycles/flop × 524 K flops = ~67 µs/call of fixed overhead.

2. **Get IPC from `perf stat`.** Pin to a P-core with `taskset -c 0`. If IPC is identical at small and large sizes, the kernel itself is steady-state at both — the gap is in non-kernel paths or per-call setup. (mgemm: IPC = 1.97 at s=64, 1.95 at s=256 — same.)

3. **Test one hypothesis at a time with a single-variable change**, ideally one that's structurally reversible. Don't bundle.

4. **Rule things out with ablation** (stub-to-no-op), not just optimization. Stubbing `simd_writeback` told us writeback is ~7% of total — useful even though we can't fully eliminate it.

5. **Bench at high iter counts (80+) pinned to a P-core.** Below that, run-to-run variance is ±3–5% and small effects vanish in noise. Without pinning, the kernel may migrate between P-cores and E-cores, contaminating cycle counts.

6. **Suspect noise on `(s=N, single combo)` jumps; trust `(mean over 9 combos at s=N, stdev < 1%)` for structural effects.**

7. **Verify sub-1.0× ratios in the C perf harness before chasing them as
   kernel-codegen gaps.** The fypp-generated Fortran bench can manufacture
   5–10% phantom gaps from binary layout alone — when the linker spreads
   overlay's kernel ~100 KB from migrated's, iTLB churn on each call
   eats ~5–10% throughput. The `tests/blas_parallel/perf/target_<name>/perf_*.c`
   harness (CMake wires `-ffunction-sections -Wl,--gc-sections` per
   executable) collapses overlay's footprint to the same few KB as
   migrated and reports honest GF/s. Procedure: if the Fortran bench
   reports <1.0×, (a) re-measure with the C perf harness; if still
   <1.0×, (b) diff inner-loop disassembly against migrated. No insn-count
   or IV-folding difference = harness overhead, not kernel codegen.
   See Addendum 14 for the full diagnosis.

---

## Hardware-specific gotchas on this machine

- **Hybrid CPU**: `perf stat` reports `cpu_atom` (E-core) and `cpu_core` (P-core) separately. Threads can migrate; pin with `taskset -c 0` (or 1, 2, 3 — P-cores 0–3, E-cores 4–7).
- **P-core L2 = 1.25 MB**. Adaptive-MC targets should aim for ~768 KB working set to leave headroom for Bp + code/stack.
- **No AVX-512** on the U-skus — AVX2 + FMA is the ceiling. SIMD-DD kernels use ymm registers (16 available).
- **x87 long-double FPU** is the only path for `long double` and `_Complex long double`. 8-deep register stack. No SIMD. ~10-cycle fmul latency → 2 independent FMA chains saturate it.
- **System `perf` mismatched** with running kernel — use `/usr/lib/linux-tools/6.8.0-111-generic/perf` (any close-enough version works; some events show `<not supported>`).

---

## Open items / things worth trying later

These weren't tested but might be worth a controlled bench:

1. **Specialized small-size kernel for mgemm s=64**: fully unrolled MNK=64 with no blocking, possibly compile-time generated. Target: close the s=64 vs s=128 gap (currently 2.5 vs 3.6 GF/s).
2. **MR=2, NR_PAN=2 shape** for mgemm: same tile area as MR=4, NR_PAN=1 but different shape. Different register pressure profile (8 acc + 2 broadcasts + 4 B-loads = 14 ymm), might suit some access patterns.
3. **Software prefetching** for Ap/Bp in inner kernel: OpenBLAS does this; we don't. May help at s≥256 where data starts to exceed L2.
4. **Bench mgemm/wgemm at OMP=4 or 8** — adaptive MC and persistent buffers may behave differently with cross-thread cache contention.
5. **Egemm + ygemm at OMP>1** — current numbers are OMP=1. Threading is the actual point of the overlay.

---

# Addendum: kind10 L3 (symm/hemm/syrk/herk) round — 2026-05-15

A second optimization pass on the symmetric and Hermitian L3 routines, after the GEMM/TRSM work above. Targets: esymm, ysymm, yhemm, esyrk, ysyrk, yherk. Pattern: diagonal-block scalar kernel plus an off-diagonal trailing `[ey]gemm` call per block-pair, blocked over column panels.

The original baseline had several catastrophic outliers — esyrk U/N s=64 at 0.24×, yhemm L/L s=64 at 0.43×, ysymm L/U at 0.75–0.95× — buried inside otherwise-passable mean speedups (1.0–1.3×). Three structural fixes, applied as a recipe to each file, cleared all of them. Plus one negative result and one "no further closure possible" finding worth recording.

## TL;DR results (OMP=1, 40 iters, P-core)

| routine | mean before | mean after | min before | min after |
|---------|-------------|------------|------------|-----------|
| esymm   | 1.16× | **1.41×** | 0.85 | 0.89 |
| ysymm   | 1.00× | **1.20×** | 0.75 | 1.09 |
| yhemm   | 0.99× | **1.18×** | **0.43** | 0.93 |
| esyrk   | 1.22× | **1.31×** | **0.24** | 0.85 |
| ysyrk   | 1.06× | **1.12×** | 0.82 | 0.99 |
| yherk   | 1.05× | **1.13×** | 0.80 | 0.97 |

## What worked

### A. Stride-1 A access in the diagonal kernel (symm/hemm)
The Netlib reference DSYMM/ZSYMM/ZHEMM walks A column-major — for SIDE='L' UPLO='L', `A(K,I)` with `K>I` reads the stored lower triangle stride-1 in K. Our overlay was reading `A_(i, k)` (row i, column k) — the *symmetric counterpart*, also stored, but **stride-lda in k** at fixed i. Every inner-iteration A load crossed a cache line.

Fix: reverse the i loop direction (backward for UPLO='L', forward for UPLO='U') so the inner k loop reads `A_(k, i)` — same column, varying row, stride-1.

For Hermitian: the conjugate moves from the cj[k] update onto the temp2 accumulation when iteration direction flips (since `conj(A(i,k)) = A(k,i)` for the stored half). Same total ops, same result, stride-1.

Impact on its own: ysymm L/U 0.75–0.95 → 1.12–1.18; yhemm L/L 0.43 → 0.93 at s=64.

### B. Single-block fast path (symm/hemm)
At small M (≤ nb), the framed `for(jc) → for(ic) → diag_kernel + gemm` path runs exactly one ic iteration with no gemm call. The framing — `omp_get_max_threads`, OMP pragma runtime check, function call into the static diag kernel, jc/ic outer loops — adds ~5–10% overhead on top of an irreducible scalar inner loop.

Fix: early fast path inside the entry function that inlines the Netlib loop directly, folds BETA into the diagonal write, and skips all framing.

Modest additional gain (~3–5%) and removes the BETA pre-scale separate pass.

### C. Smaller `nb` cap (all six)
Default `nb=64` (esymm/esyrk) or `64–256` (ysymm/yhemm/ysyrk/yherk). At s=128 with nb=64, the entire problem is a single diagonal block — zero gemm trailing-update work, 100% of the time in the scalar diag kernel. The gemm-driven speedup that justifies this overlay never fires.

Dropped to:
- **esymm: cap 128 → 64**. egemm at K=32 doesn't amortize for real long-double (packing overhead too high), so the floor stays at 64.
- **ysymm/yhemm/ysyrk/yherk: cap 256 → 32**. Complex long-double per-element work is heavy enough that ygemm tolerates K=32; packing fraction is small.
- **esyrk: cap stays 64** (same real long-double reasoning).

Impact: s=128 and s=256 fragment into 2×2 and 4×4 block grids with the trailing `[ey]gemm` calls picking up most of the off-diagonal work.

### D. `restrict` pointers on all A/B/C parameters (largest single per-file win)
C `const T*` doesn't tell gcc the buffers don't alias. The diag inner loop is `cj[k] += temp1 * ai[k]; temp2 += bj[k] * cconj(ai[k]);` — gcc has to assume the `cj[k]` store might have changed `ai[k]` or `bj[k]`, so it reloads them from memory on every iteration. Verified by objdump: `fldt (a.real); fldt (a.imag); ... fstpt (cj.real); fstpt (cj.imag); fldt (a.real); fldt (a.imag);` — three redundant loads per iteration.

Adding `restrict` to `a`, `b`, `c` on both the entry-point signatures (yhemm_, ysymm_, esymm_, esyrk_, ysyrk_, yherk_) and the static helper signatures eliminates the reloads. The BLAS spec already forbids aliasing, so this is a free correctness assertion.

Impact: ysymm L/L 1.04 → 1.09, yhemm L/L 0.93 → 0.96, ysymm L/U all sizes +1–2%. Biggest on the complex routines where the inner loop has to keep both real/imag halves of A live across the C store. esymm (real long-double) was flat — a single scalar write doesn't trigger the same reload chain.

## What didn't work

### Manual `__real__`/`__imag__` real-arithmetic expansion (yhemm)
After noticing that `bj[k] * cconj(ai[k])` compiles to ~25 x87 ops (verified by objdump), tried rewriting the inner loop as explicit scalar `long double` arithmetic:

```c
__real__ cj[k] += t1r * ar - t1i * ai;
__imag__ cj[k] += t1r * ai + t1i * ar;
t2r += br * ar + bi * ai;
t2i += bi * ar - br * ai;
```

**Result**: regressed L/L 64 from 0.93× → 0.86×. Disassembly showed gcc had successfully eliminated the redundant A/B loads, but spilled `t1.i` to the stack (`fldt 0x10(%rsp)`) instead and reloaded it twice per iteration. The x87 stack is 8 slots and the expanded form needs to keep more values live than the implicit form. Net: more loads, slower.

`-fcx-fortran-rules` already gets gcc to inline complex multiplication (no `__mulxc3` libcall), so the manual expansion buys nothing the compiler isn't already doing.

### gcc scheduler flags `-fschedule-insns -fsched-pressure` (yhemm)
Both default-off on x86 due to historic x87 quirks. Tried enabling them per-file via `set_source_files_properties`. Result: small regression (L/L 64: 0.93 → 0.89). Tried `-fmodulo-sched` separately: net negative too. The default scheduler is already doing the best it can on this inner loop.

## Findings worth recording

### x87 codegen ceiling on `_Complex long double` (yhemm L/L)
Even with all fixes applied, yhemm L/L sits at 0.92–0.96× across all sizes — overlay ~2.15 GF/s, migrated ~2.30 GF/s. The 7% gap is a hard floor: gcc and gfortran are producing **the same inlined-cmul + fchs-conjugate inner-loop structure** (verified via objdump on both `.o` files); the difference is x87 register scheduling, which gfortran does slightly better on this specific pattern.

Closing the gap would require:
- **Type-level rewrite**: SoA (`long double` re/im arrays) instead of `_Complex long double`. The Fortran ABI forbids changing the public signature, so this only works as a local repack into scratch buffers inside the kernel — and the ~2 KB unpack/repack per 32×32 diag block likely eats the gain.
- **Inline x87 assembly**: technically feasible but gcc's extended asm with x87 constraints is famously buggy; you'd be chasing gfortran's already-optimal schedule with a more fragile encoding.

Not pursued. The remaining 7% is below the noise floor of any realistic OMP>1 win.

### The `nb` choice splits real vs complex
Underlying principle behind fix C above: each complex long-double op is ~8× heavier than scalar long-double, so the packing fraction of total work shrinks and ygemm pays off at K=32. Real long-double's lighter per-op work means egemm needs larger K (≥ 64) before packing amortizes — the scalar inner kernel has enough x87 ILP that gemm-via-pack isn't a win until then.

### Aliasing pessimism is real and `restrict` is free
This is the most important takeaway from the L3 round. Three lines of code (`restrict` on a, b, c at the entry-point + helper) closed several percentage points of margin across complex routines. The disassembly proof made this obvious — but it would have been easy to miss had I not been looking at the actual generated code.

Whenever a BLAS-shape kernel has the inner-loop pattern `c[i] += stuff(a[i], b[i])`, `restrict` on those three pointers is almost always a free win on x87 long-double — because the compiler can otherwise assume the C write may have changed A or B.

## Additions to "diagnostic methodology"

7. **When the overlay loses on a single (uplo, trans, size) cell, check whether the framed path is hitting a *single-block regime* — i.e. nb is large enough that the gemm trailing-update never fires.** That's almost always where the catastrophic outliers live (esyrk U/N 0.24×, yhemm L/L 0.43×, ytrmm L/U/N/N 0.45×). The fix is rarely "improve the diag kernel" — it's "shrink nb so the trailing gemm picks up the slack."

8. **For x87 long-double kernels, always check the objdump for redundant `fldt` after a `fstpt` to A/B addresses.** Two reloads of the same memory location per inner iteration is a tell that the compiler is being conservative about aliasing. `restrict` fixes it.

9. **Negative results (this round): two compiler flag experiments and one source-level manual expansion** all regressed. Save effort: prove the disassembly is suboptimal *before* trying to fix it, and verify the fix changes the disassembly in the expected direction *before* benching.

10. **Compare migrated to the theoretical x87 ceiling before "fixing" the overlay.** Real long-double dot product has three regimes:
    - Single-accumulator: ~1.3 GF/s (7-cycle fadd dep chain)
    - 2 independent accumulators: ~2.7 GF/s
    - Fully pipelined fmul+fadd, no chain: ~9.4 GF/s (rarely achievable)

    Migrated DSYRK at TR='T' s=64 hits 2.36 GF/s — that's right at the 2-acc ceiling. **If migrated is at a known x87 ceiling, you can't beat it; you can only match it by replicating the same dep-chain count.** Compute migrated's % of peak first; if it's ≥85% of some ceiling, identify which ceiling and replicate.

## K-axis 2-unroll: works on real dot products, breaks on complex ones

Counterpart to the documented ygemm dot-product regression (§7 above). The pattern:

```c
T s0 = 0.0L, s1 = 0.0L;
int l = 0;
for (; l + 1 < K; l += 2) {
    s0 += Ai[l]     * Aj[l];
    s1 += Ai[l + 1] * Aj[l + 1];
}
T s = s0 + s1;
for (; l < K; ++l) s += Ai[l] * Aj[l];
```

**Real long-double (esyrk TR='T')**: cleanly doubles throughput on a serial fadd-chain kernel. esyrk L/T 64: 0.85× → 0.98×; every other L/T,U/T cell pinned to ≥0.98×. Matches gfortran's reference DSYRK rate.

**Complex long-double dot product paths (ygemm TN/CN, CT/CC; ysyrk/yherk T/C)**: regressed under the same pattern, or already at parity so the unroll buys nothing. ygemm bench: TN/CN 1.00× → 0.80×; CT/CC 0.99× → 0.35–0.56×. Two reasons:
1. Complex multiplication is itself ~4× heavier than real, so the single-chain bottleneck is less of the runtime (chain latency hidden by cmul work). gcc's scalar single-acc loop was already well-scheduled; the split disrupted its register allocation.
2. ygemm had a `conj_b ? ~bv0 : bv0` conditional inside the loop body that wasn't constant-folded; the K-unroll put a runtime branch in the critical path. Even without that, naïve K-unroll on complex would help less.

**Rule of thumb:** K-axis 2-unroll is the right structural move on **real scalar long-double dot-product reductions**. For complex types, leave the inner loop single-accumulator — gcc's scheduler does better when there's only one cmul chain to lay out on the x87 stack.

## Addendum 2: ytrmm nb 64 → 32 + etrmm at codegen ceiling — 2026-05-15

Two follow-on findings from the trmm round.

### ytrmm L U N N s=64: 0.45× → 1.10× via nb 64 → 32 (single-block-regime fix)

Same failure mode documented for ysymm/yhemm/ysyrk/yherk in Addendum 1, now confirmed on ytrmm. At nb=64, the gate `M >= 2*nb` falls through to the unblocked scalar core at M=64; the trailing ygemm never fires. Dropping default nb to 32 engages the blocked path: two 32×32 diag kernels + one 32×64×32 ygemm. ytrmm L U N N s=64 went 0.42 → 2.20 GF/s.

Distribution-level: median 0.99× → 1.08×, min 0.45× → 0.95×, sub-parity 14/27 → 8/27. All remaining cells within 5% of parity.

**Generalization:** the "shrink nb for complex variants" recipe applies to **every** complex L3 routine on this overlay — symm, hemm, syrk, herk, and trmm — for the same reason (ygemm at K=32 amortizes well; real long-double doesn't). Apply by default; don't wait for a catastrophic outlier to spot it.

### etrmm sub-parity cells are not a codegen gap

Disassembled `trmm_llt._omp_fn.0` (overlay) and the corresponding Fortran inner loop in `etrmm.f.o` (migrated). The inner dot-product loop is **byte-identical**:

```
fldt   0x10(%rXX,%rYY,1)   ; load A(k,i)
fldt   0x10(%rXX,%rYY,1)   ; load B(k,j)
add    $0x10,%rYY
fmulp  %st,%st(1)
faddp  %st,%st(1)
cmp    %rYY,%rZZ
jne    ...
```

6 instructions, identical encoding. gcc -O3 + `-fcx-fortran-rules` produces the same x87 instruction sequence as gfortran for the canonical real long-double dot-product kernel.

**Implication:** for any kind10 real routine whose inner loop is this 6-instruction sequence, the overlay can match migrated but cannot beat it via codegen. Sub-parity cells in this regime (etrmm's 0.91–0.99× cluster) are framing overhead or run-to-run noise, not a fixable codegen gap.

**Practical rule:** before optimizing a real long-double scalar inner loop, disassemble both overlay and migrated. If the inner loops are byte-identical, stop — chase outer-loop framing or accept noise. Combined with methodology #10 (compare migrated to ceiling), this gives two independent ways to detect "no win possible" before sinking iter budget.

## Addendum 3: `omp_get_max_threads()` per-call overhead — 2026-05-16

`eger` was reported sub-parity vs migrated at OMP=1 (0.87× across all sizes). Inner-loop disassembly was byte-identical to the migrated reference (the "no codegen gap" condition from Addendum 2). The overhead was per-call, not per-iteration.

### Root cause

Every overlay using OMP had the pattern:

```c
#ifdef _OPENMP
    const int use_omp = (N >= EGER_OMP_MIN && omp_get_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
```

`omp_get_max_threads()` is a libgomp function call — not an intrinsic, not inlined. Each BLAS invocation paid ~15–50 ns to ask whether the runtime had >1 threads. The result never changes during a process's lifetime (the OpenMP spec allows it to change but BLAS callers never do, and the migrated reference doesn't honour mid-run changes either).

At small N where the actual FP work is microseconds, this single call was 10–15% of total runtime.

### Fix

Cache the value in a relaxed-atomic static. New header at `src/parallel_blas/common/blas_omp.h`:

```c
static inline int blas_omp_max_threads(void)
{
    static int cached = 0;
    int v = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (__builtin_expect(v == 0, 0)) {
        v = omp_get_max_threads();
        if (v < 1) v = 1;
        __atomic_store_n(&cached, v, __ATOMIC_RELAXED);
    }
    return v;
}
```

Hot path is a single .bss load. The benign race on cache init is fine — every writer stores the same value.

### Result (sweep applied to all 90 overlays)

Aggregate mean speedup vs migrated, OMP=1 / OMP=4 (size=256, iters=3):

| target       | OMP=1 before / after | OMP=4 before / after |
|--------------|----------------------|----------------------|
| kind10       | 1.10× / **1.20×**    | 2.01× / 2.16×        |
| kind16       | 0.98× / **1.01×**    | 1.80× / 1.84×        |
| multifloats  | 8.15× / **8.47×**    | 13.94× / 14.52×      |

Headline per-routine wins at OMP=1:

- `wmscal`  14.42× → 21.48× (+7.1)
- `ycopy`    3.49× →  6.99× (+3.5)
- `maxpy`   10.40× → 12.99× (+2.6)
- `wscal`    9.47× → 11.58× (+2.1)
- `xcopy`    0.60× →  2.00× (+1.4)

Routines previously sub-parity that returned to ≥1.0×: `eaxpy`, `ecopy`, `edot`, `eger`, `eswap`, `ydotu`, `qcopy`, `xcopy`, `xswap`. The single overlay we proved this on (`eger`) went 0.87× → 0.99× at N=256.

### Generalizations / rules

1. **Never call libgomp helpers (`omp_get_max_threads`, `omp_get_num_procs`, `omp_get_dynamic`) on the hot path.** They're real function calls into a shared library. Cache or hoist anything you query once per process.

2. **The `if()` clause on `#pragma omp parallel for` is not free even when false.** GOMP_parallel still gets called and decides not to fork. For routines where the serial path is the common case (N below OMP threshold), branching around the pragma entirely is faster — but only if the loop body can be kept inline. Factoring the loop body into a separate function for code-sharing hurt eger's scalar x87 inner loop noticeably (lost the implicit `t` register reuse across iterations).

3. **Per-call overhead is the right thing to chase once codegen is at parity.** Once the inner loop is byte-identical to migrated (Addendum 2 rule), the only remaining variable is what the caller pays *before* and *after* the loop. The libgomp call was 1 instruction in the source but 50 ns in practice — invisible to anyone who only reads C, obvious in the disassembly.

4. **Apply OMP-overhead fixes broadly, then re-bench.** A handful of routines showed small (<10%) regressions in the after-sweep doc. Several of those (`escal`, `esyr2`, `mspr2`, etc.) sit in source files the sweep didn't touch — they're `BLAS_BENCH_ITERS=3` sampling noise, not real regressions. With this many routines × this many shapes, individual-cell variance is large; trust the aggregate means.

## Addendum 4: libm function calls in the inner expression — 2026-05-16

Reported: `yrotg` (kind10 complex Givens generator) at 0.48× of migrated — by far the worst speedup in the entire overlay after the OMP-sweep round.

### Diagnosis

Inner loops were already byte-identical to migrated for the rotg family (single-call routines, no inner loop to compare). The cost had to be *inside* the body. One-line objdump check:

    overlay  yrotg.c.o:   3 × call hypotl@plt
    migrated yrotg.f90.o: 0 external calls

`hypotl(x, y)` returns sqrt(x² + y²) with overflow protection — for long double on glibc that's a software implementation, ~100–300 cycles per call. Three of them per yrotg invocation = the entire missing 50% of throughput.

The migrated Fortran implements Anderson 2017 ("Safe Scaling in Level 1 BLAS") with direct algebra on `ABSSQ(t) = Re(t)² + Im(t)²`, branching to a scaled path only when inputs risk overflow or underflow. The common-input fast path uses two inline `fsqrt`s and otherwise stays in registers.

### Fix

Replaced the three-hypot pattern with the same unscaled algebra:

```c
g2 = br*br + bi*bi;             // |b|²
if (g2 == 0)  { *c = 1; *s = 0; return; }
f2 = ar*ar + ai*ai;             // |a|²
if (f2 == 0)  { *c = 0; /* sqrt(g2), conj(b)/|b| */ return; }
h2 = f2 + g2;
*c  = sqrtl(f2 / h2);            // |a|/sqrt(|a|²+|b|²)
*a_ = a / *c;                    // sign(a)·sqrt(|a|²+|b|²)
d   = sqrtl(f2 * h2);
*s  = conj(b) * (a / d);
```

Two `sqrtl` calls (each compiles to one `fsqrt` instruction in this build) and otherwise direct arithmetic.

Speedups (OMP=1, BLAS_BENCH_SIZES=10000 looped per call):

| target          | routine | before | after |
|-----------------|---------|--------|-------|
| kind10          | yrotg   | 0.48×  | **1.78×** |
| kind16          | xrotg   | 0.77×  | 1.21× |
| multifloats     | wrotg   | 3.42×  | **5.88×** |

multifloats had the biggest absolute jump because `cabsdd` (the DD analog of `hypotl`) is itself DD arithmetic — eliminating two of those nets more cycles even relative to the already-DD `sqrtdd`.

### Generalization

Three categories of hidden function calls to look for during overlay diagnosis, sorted by cost on this hardware:

1. **libm long-double helpers** (`hypotl`, `sinl`, `cosl`, `powl`, `cbrtl`, `frexpl`, `ldexpl`): ~100–300 cycles each. None are inlined; `-ffast-math` doesn't help most of them. The Fortran reference often avoids these entirely via direct algebra — so a Fortran-to-C migration that "preserves intent" by reaching for libm reintroduces a cost the original didn't have.
2. **libquadmath analogs** (`hypotq`, `sqrtq`, `sinq`, etc.): higher absolute cost than libm long-double because `__float128` arithmetic is itself emulated. Sweep equivalent.
3. **libgomp helpers** (`omp_get_max_threads`, `omp_get_num_procs`): ~15–50 ns per call. Addendum 3 covered these.

**Diagnostic rule:** when an overlay is sub-parity and the inner loop body has been verified equal to migrated, run `objdump -dr` and look at the `R_X86_64_PLT32` relocations. The list of names is the list of suspects, and `sqrt` is almost always the only one that's actually intrinsic. If `hypot`/`pow`/etc. appears, see if the source-level algorithm can be rewritten to avoid it — the BLAS/LAPACK reference frequently has already done so.

**Practical rule for porting:** before introducing `hypot`/`pow`/`exp` in a C overlay, check whether the original Fortran source actually uses the equivalent intrinsic. If it doesn't, find the algebraic identity it used instead and port that.

## Addendum 5: algorithmic gap — Blue's single-pass nrm2 — 2026-05-16

Sweep follow-up from Addendum 4. After clearing the avoidable libm calls in the L2 norm routines (qnrm2/qxnrm2 finiteq → __builtin_isfinite, fabsq → __builtin_fabsf128), `qnrm2` and `qxnrm2` were still at 0.39–0.40× of migrated. `enrm2` and `eynrm2` were at 0.84–0.88×. The remaining gap was **algorithmic, not call-overhead**.

### Diagnosis

The overlays used the obvious **two-pass scaled** L2 norm:

```c
scale = 0;  for i: scale = max(scale, |x[i]|)   // pass 1
s = 0;      for i: t = x[i]/scale; s += t*t     // pass 2
return scale * sqrt(s)
```

Two trips over X. For __float128 every element comparison and multiply is a libgcc soft-float call (`__lttf2`, `__multf3`, `__divtf3`) — doubling the trip count doubles the call count. For x87 long double the second pass is pure extra memory traffic and fmul/fadd latency.

The migrated Fortran (kind16/qnrm2.f90) implements **Blue's algorithm** (Anderson 2017, "Safe Scaling in the Level 1 BLAS"; Blue 1978, "A Portable Fortran Program to Find the Euclidean Norm of a Vector"). Three magnitude-bucketed accumulators (`abig`/`amed`/`asml`) each accumulate the sum of squares within its own scale range, so the whole vector is processed in a **single pass**:

```c
for i:
   ax = |x[i]|
   if ax > btbig   abig += (ax * bsbig)²    // scale down before squaring
   elif ax < btsml asml += (ax * bssml)²    // scale up before squaring
   else            amed += ax²              // unscaled
// combine abig/amed/asml at end
```

The thresholds (`btsml`/`btbig`/`bssml`/`bsbig`) are radix-power constants chosen so the scaled squares can't overflow or underflow.

### Fix

Ported Blue's algorithm directly to C for all four nrm2 routines:

- Constants computed once per process via `scalbnq(1.0Q, k)` / `ldexpl(1.0L, k)` in a `__attribute__((cold))` static-init function.
- Three accumulators in registers; single pass over X.
- Complex variants (`eynrm2`, `qxnrm2`) factor the bucketing into a `static inline` helper called twice per element (Re, Im).

Speedup vs migrated (OMP=1, N=4096):

| target  | routine | before | after  |
|---------|---------|--------|--------|
| kind16  | qnrm2   | 0.39×  | 0.99×  |
| kind16  | qxnrm2  | 0.40×  | 1.00×  |
| kind10  | enrm2   | 0.88×  | **1.98×** |
| kind10  | eynrm2  | 0.84×  | **1.97×** |

kind10 ends up *beating* migrated by ~2× even though the algorithm is the same. Two reasons:

1. The migrated has a `if (.not. blue_initialized)` branch in the hot path. Our C version uses `__builtin_expect(.. == 0, 0)` so the branch predicts perfectly after the first call.
2. gcc with `-O3` schedules the three-accumulator bucketing loop better on x87 than gfortran does on the same source — both produce identical fmul/fadd op counts but gcc emits fewer fxch/fld redundancies for this specific shape.

multifloats `mnrm2`/`mwnrm2` were not changed: they already run at ~20× via explicit AVX2+FMA3 SIMD kernels. The algorithm-level Blue port is orthogonal to the SIMD optimization and would interfere with the vectorized inner loop.

### Rules

1. **When the libm-call sweep doesn't close the gap, the algorithm is wrong.** Once `objdump -dr` shows only fundamental ops (`__multf3`, `__addtf3`, etc. for kind16; `fmul`, `fadd` for kind10), and the inner loop is still slower than migrated, the suspect shifts from per-call cost to per-pass cost. Count how many times the input vector is read.

2. **The Fortran reference often implements the better algorithm.** The BLAS/LAPACK reference has accumulated half a century of numerical-stability work; the "obvious" port of `nrm2` to "find max, then sum" loses to the reference for both speed and safety. Read the Fortran source before writing the C overlay.

3. **Single-pass beats two-pass even before SIMD considerations.** Memory traffic matters even for compute-bound kernels at small N; for L1 routines (memory-bound at any N) it's the dominant cost. If you can fold "find scaling factor" and "sum scaled" into one pass via bucketing or running max, do it.

4. **Process-once init costs nothing in steady state.** Blue's algorithm needs four scale-threshold constants. Computing them lazily via `static int inited = 0` + `if (__builtin_expect(!inited, 0)) ...` adds two cycles to the post-init hot path (a load and a predicted-taken branch). The Fortran reference SAVEs these — equivalent.

---

## Addendum 6: noise vs. genuine sub-parity (larger-N re-bench, 2026-05-16)

The N=256 default in `blas_overlay_report.py` is noise-prone — the timer
resolution is ~1µs and an L1 routine at N=256 runs in tens of µs, so a
single GC blip or context switch flips the speedup by 0.2×. The survey
flagged 13 routines as sub-parity (< 0.95×) at N=256. Re-benching at
N where the kernel runs ≥10 ms separates measurement noise from real
gaps.

### Re-bench protocol

| family   | N tried              | iters | warmup |
|----------|----------------------|-------|--------|
| L1       | 4096, 16384, 65536   | 5     | 2      |
| L2 dense | 512, 1024, 2048      | 5     | 2      |
| L2 banded| 512, 1024, 2048 (K=32)| 5    | 2      |
| L3       | 256, 512, 1024       | 5     | 2      |
| L3 sym   | 128, 256, 512        | 5     | 2      |

### Results

| target  | routine  | N=256 (old) | larger-N median | verdict        |
|---------|----------|-------------|------------------|----------------|
| kind16  | qswap    | 0.50×       | 1.00×            | noise          |
| kind10  | ygemmtr  | 0.80×       | 1.00×            | noise          |
| kind10  | esyr2    | 0.87×       | 1.05×            | noise          |
| kind10  | esbmv    | 0.91×       | 1.00×            | noise          |
| kind10  | yhemm    | 0.89×       | 0.90× (LL) / 1.2-1.4× (others) | noise (LL slightly behind) |
| kind10  | ydotc    | 0.78×       | 0.86–0.90×       | **persistent** (x87 stack, documented) |
| kind10  | egemv    | 0.84×       | 0.74–0.85× (N), 1.05–1.22× (T) | **persistent** on N path |
| kind10  | espr     | 0.85×       | 0.78–0.90×       | **persistent** |
| kind10  | yscal    | 0.89×       | 0.93×            | persistent (x87 stack) |
| kind10  | yescal   | 0.87×       | 0.92–0.93×       | persistent     |
| kind10  | erot     | 0.80×       | 0.93×            | persistent     |
| kind10  | erotm    | 0.89×       | 0.93×            | persistent     |
| kind16  | qsymm    | 0.91×       | 0.88–0.92× (L*)  | **persistent** |

5 of 13 routines were pure noise at N=256. 6 settle at 0.92–0.93× —
the documented x87-overhead floor (every `*=` on a packed
long-double[] pays an `fld m80` + `fstp m80` cycle that the migrated
reference avoids by working out of registers across a wider unroll).
2 are real algorithmic gaps:

- **egemv N-path** sits at 0.74–0.85× across all sizes. The migrated
  reference does the matrix-vector product with an inner `dot` accumulator;
  our overlay does an outer SAXPY pattern that thrashes the destination
  vector. Worth a rewrite.
- **espr** (kind10 real symmetric rank-1 update) at 0.78–0.90×. The
  reference uses a column-walking pattern; our overlay walks rows and
  pays for x87 load/store on every diagonal element.

### Rule

**N=256 is below the noise floor for L1 routines.** When the
overlay-report flags a routine as sub-parity, re-bench at N where the
kernel takes ≥10 ms before drawing any conclusions. The "13 sub-parity
routines" the original survey reported was really 8 — and of those, only
2 are worth attacking; the rest are at the documented x87 floor.

---

## Addendum 7: gfortran's shared-index pointer-walk codegen (2026-05-16)

### The gap

`egemv` N-path (`y := alpha · A · x + beta · y`, `A` is M×N stored column-major)
sat at 0.74–0.85× vs migrated. Identical algorithm, identical instruction
count (13 per inner-loop iter), identical L1-dcache miss count
(measured via `perf stat`). Yet 17% more total instructions, ~9% more
cycles.

### The diagnosis

The overlay's 2-way j-unrolled inner loop:

```asm
110:  lea (%r8,%r12,1),%r13     # recompute A0 base from A1 base
114:  add $0x10,%rdx             # y ptr += 16
118:  add $0x10,%r12             # A1 ptr += 16
11c:  fldt 0x0(%r13,%rdi,1)     # load A0[i]
...
134:  cmp %rdx,%r9
137:  jne 0x110
```

gcc maintains **three independent pointers** (Y, A0, A1) and emits a
`lea` + two `add`s per iter to advance them. That's 3 of the 13 insts
per iter just managing pointer arithmetic.

gfortran's reference DGEMV — also 2-way j-unrolled by `-O3` auto-unroll
— emits a different shape:

```asm
820:  fldt 0x10(%rdi,%r9,1)     # load A0[i] via shared index %r9
825:  fmul %st(2),%st
827:  fldt (%r15,%r9,1)         # load Y[i] via same %r9
82b:  faddp
82d:  fldt 0x10(%rsi,%r9,1)     # load A1[i] via same %r9
832:  fmul %st(2),%st
834:  faddp
836:  fstpt (%r15,%r9,1)        # store Y[i] via same %r9
83a:  add $0x10,%r9             # ONE pointer increment
83e:  cmp %r11,%r9
841:  jne 0x820
```

**Single shared index register `%r9` for all three loads, with three
different base registers.** One `add` per iter. 11 insts per iter
instead of 13.

### Why gcc didn't emit the same shape

Tried four variants of the C source:

1. `for (int i = i_lo; i < i_hi; ++i) y[i] = ... a0[i] ... a1[i]` —
   gcc keeps three pointers (13 insts).
2. Same as #1 with `restrict` on all three pointers — same (13 insts).
3. `size_t` induction variable — same (13 insts).
4. **Explicit byte-offset walk** through three `char*` bases —
   gcc emits the shared-index pattern (11 insts).

The "what should be obvious" version (1) doesn't trigger gcc's
shared-index loop transform. Only the byte-cast form does:

```c
char *restrict yp = (char *)(y + i_lo);
const char *restrict a0 = (const char *)&A_(i_lo, j);
const char *restrict a1 = (const char *)&A_(i_lo, j + 1);
const size_t end = (size_t)span * sizeof(T);
for (size_t k = 0; k < end; k += sizeof(T)) {
    T *yk        = (T *)(yp + k);
    const T *a0k = (const T *)(a0 + k);
    const T *a1k = (const T *)(a1 + k);
    *yk = (*yk + t0 * *a0k) + t1 * *a1k;
}
```

### Results

| metric           | before | after  | migrated |
|------------------|-------:|-------:|---------:|
| insts/inner iter | 13     | 11     | 11       |
| insts (N=1024×50)| 375M   | 320M   | 320M     |
| cycles (N=1024×50)|340M   | 310M   | 311M     |
| GF/s (microbench)| 1.45   | 1.64   | 1.64     |
| speedup (N=512)  | 0.74×  | 0.88×  | —        |
| speedup (N=1024) | 0.84×  | 0.91×  | —        |
| speedup (N=2048) | 0.85×  | 0.92×  | —        |
| speedup OMP=4 N=512 | —   | 2.73×  | —        |

Microbench (no OMP wrapper) hits exact parity. The remaining ~0.08
gap in the bench harness is fixed-cost OMP outlining (firstprivate
struct pack, `__GOMP_parallel` call, omp_fn prologue) that fires even
when `use_omp=false`.

### Rules

1. **Same loop shape can produce different codegen.** Instruction count
   per iter depends on whether gcc folds multiple pointer increments
   into a single shared-index walk. Default codegen for
   `array[i] = ... other[i] ... third[i]` keeps three pointers; only
   the byte-offset walk through `char*` bases triggers the fold.

2. **Always count insts in `objdump`, not in C.** A 13-line C loop and
   an 11-line C loop can compile to the same number of x86 insts; a
   "natural" C loop can be 17% more insts than a structurally-identical
   Fortran loop. The `objdump -d` count is the only number that
   matters for x87 perf at this level.

3. **`perf stat -e cycles,instructions,L1-dcache-load-misses` is the
   diagnostic.** If misses are identical and IPC is similar, the gap is
   instruction count. If insts are identical and cycles diverge, the
   gap is microarchitectural (port contention, branch prediction,
   cache associativity). The two cases call for different fixes.

4. **The byte-offset walk is the workaround for gcc's missed
   shared-index transform.** Ugly but correct (well-defined via the
   `char*` alias rule + `restrict`). Apply when:
   - inner loop walks ≥2 arrays in lockstep with the same stride;
   - operand size is large (long double / __float128) so per-iter
     pointer increments are a significant fraction of inner-loop insts;
   - `-O3` auto-unroll hasn't already done the work (kind16/multifloats
     may differ).

---

## Addendum 8: pointer-walk loop pattern for L2 packed kernels (2026-05-16)

`espr` (kind10 real symmetric packed rank-1, `A := alpha·x·xᵀ + A`)
benched at 0.78–0.93× across (uplo, size). Same diagnosis pattern as
Addendum 7: walk one pointer + walk a counter = 10 insns per inner
iter; walk two pointers and compare = 9.

### Before vs after

The natural C form:

```c
for (int i = 0; i <= j; ++i) ap[kk + i] += x[i] * tmp;
```

compiles to 10 insns per iter (one pointer increment + one counter
inc + one cmp). Rewritten as a pointer-walk:

```c
T *restrict apk  = &ap[kk];
T *restrict aend = apk + j + 1;
const T *restrict xp = x;
for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
```

compiles to 9 insns per iter — two pointer increments + cmp, no
separate counter. Matches gfortran reference DSPR codegen.

### Results

| uplo | N    | before | after |
|------|------|-------:|------:|
| L    | 512  | 0.78×  | 1.01× |
| L    | 1024 | 0.83×  | 1.09× |
| L    | 2048 | 0.86×  | 1.01× |
| U    | 512  | 0.87×  | 0.99× |
| U    | 1024 | 0.87×  | 0.97× |
| U    | 2048 | 0.93×  | 0.97× |
| L OMP=4 | 1024 | 1.04× | **2.09×** |
| U OMP=4 | 1024 | 1.05× | **1.67×** |

Fuzz is bit-exact (max err 0.0): pointer-walk preserves the exact
operation order of the counter loop.

### Rules

5. **Default to pointer-walk loops for two-array lockstep AXPY shapes.**
   `for (int i = ...; ++i) a[i] += x[i] * t` is fine for one-array
   patterns; for two arrays in lockstep it costs one extra `inc` per
   iter that gfortran wouldn't emit. The pointer form costs more
   characters to write but matches what `-O3` would do for the Fortran
   source.

6. **Addendum 7's shared-index trick and Addendum 8's pointer-walk are
   separate codegen patches.** Apply Addendum 7 (`char*` byte-offset
   walk) when ≥3 pointers walk in lockstep with the same stride; apply
   Addendum 8 (pointer-compare loop) when 2 pointers walk in lockstep.
   gcc handles the 2-pointer case fine when the loop is written as
   pointer-compare; for 3+ pointers it still loses to fortran unless
   you go through `char*`.

---

## Addendum 9: shared-index applies to 2-pointer rotations (2026-05-16)

`erot` benched at 0.93× across all N (4096–65536). Same diagnosis: the
inner loop walks two arrays in lockstep, and gcc emits two pointer
increments instead of one shared-index walk.

| metric           | overlay | migrated |
|------------------|--------:|---------:|
| insts/inner iter | 18      | 16       |
| cycles (N=16384×200) | 51M | 48M      |
| insts  (N=16384×200) | 60M | 57M      |

Reference DROT at `-O3` uses shared `(%rax,%rdx,1)` / `(%rsi,%rdx,1)`
addressing — one increment. Applying the same `char*` byte-offset
walk from Addendum 7:

| N      | before | after |
|--------|-------:|------:|
| 4096   | 0.93×  | 0.99× |
| 16384  | 0.93×  | 1.00× |
| 65536  | 0.93×  | 1.07× |

Bit-exact under fuzz.

### Negative result: erotm doesn't benefit

Tried the same trick on `erotm`. **Both** the byte-offset walk and a
pointer-compare walk made it *slower* (0.93× → 0.87×, then 0.82× at
N=65536).

Confirmed root cause by counting `fstpt` instructions and asm bytes
in the compiled `.o`:

| variant                | fstpt count | asm lines |
|------------------------|------------:|----------:|
| `for (int i...)`       | 18          | 372       |
| `for (size_t k...)` byte-offset | 6   | 113       |

The counter form triggers gcc's **loop unswitching** pass: the
`if (flag<0) / else if (==0) / else` chain inside the inlined `step()`
helper is hoisted *outside* the loop, producing 3 specialized inner
loops (one per flag case) with zero branches in the hot loop. The
byte-offset form doesn't trigger unswitching — one inner loop with
the 3-way if/else evaluated every iteration. ~2 extra branch insns
per iter, plus mispredict cost when the predictor is cold.

Reverted to the original counter form.

### Rule

7. **gcc loop-unswitching is loop-shape-dependent.** The byte-offset
   shared-index walk wins on straight-line FP bodies but defeats
   unswitching when the body has loop-invariant branches. Symptom:
   compiling the file shrinks from N specialized loops to 1
   generic-branchy loop (count `fstpt` in `objdump`, or `wc -l` the
   disasm). When the body has a loop-invariant `if`/switch — even
   buried in a `static inline` helper — prefer the natural
   `for (int i...)` form and accept the 2-pointer-increment overhead;
   the saved branches more than pay for it. Verify by counting
   `fstpt` (or whatever store insn the body emits) in `objdump -d`
   before benching: a 3× drop in store count between variants means
   unswitching was lost.

---

## Addendum 10: the shared-index gap is OMP-specific, not general (2026-05-16)

Tested Addendum 7's premise — "gcc only emits shared-index from
byte-offset walk, not from natural `for (int i...)` C" — by compiling
isolated variants:

| compiler | OMP outlining? | natural C form    |
|----------|----------------|-------------------|
| gcc      | no             | **shared-index** ✓ (one `add`/iter) |
| gcc      | yes            | two pointers (two `add`/iter)       |
| icx      | no             | shared-index ✓ AND auto-unrolls outer loop by 2 |

The premise was wrong as stated. gcc with `-O3` *does* emit shared-index
from natural `for (int i = 0; i < n; ++i) y[i] = ... x[i] ...` C.
The Addendum-7 case (egemv) failed only because of the `#pragma omp
parallel if(use_omp) firstprivate(i_lo, i_hi)` wrapper — the OMP
outlining (capturing variables into a struct and passing through an
`._omp_fn.0` function) disables gcc's shared-index transform.

The `char*` byte-offset workaround **does** re-enable shared-index
through OMP outlining, so the egemv fix stands. But it's a workaround
for an OMP-codegen interaction, not a general C codegen limitation.

### Cleanup: erot reverted to natural form

`erot` has no OMP wrapper, so the `char*` trick was unnecessary. Reverted
to:

```c
for (int i = 0; i < n; ++i) {
    T tx = c * x[i] + s * y[i];
    y[i] = c * y[i] - s * x[i];
    x[i] = tx;
}
```

Same shared-index codegen (16 insns/iter), same bench result
(0.94–1.01×), bit-exact.

### Decision tree

When you see two `add`s per iter in `objdump` of a 2-or-3-array
lockstep AXPY loop:

1. **Is the source already `for (int i...) ... a[i] ... b[i] ...`?**
   If gcc *still* emits two pointer increments, check if the function
   sits inside an `#pragma omp parallel` block (outlined into a
   `_omp_fn` function).
2. **OMP-outlined?** → apply the `char*` byte-offset walk (Addendum 7).
   The cast is the cost of working around gcc's OMP-context loop opt.
3. **Not OMP-outlined?** → the natural form already works; the byte-
   offset trick is just visual noise. Don't add it.
4. **Source is pointer-walk** (`for (; xp<xe; ++xp, ++yp)`) → switch
   to natural `for (int i...)`. Pointer-walk form blocks gcc's
   shared-index transform even outside OMP.

### Rule

8. **Workarounds need to point at what they work around.** "gcc can't
   do X" is a different claim from "gcc can't do X under OMP
   outlining". When the workaround is ugly (here, a `char*` cast),
   write the surrounding comment so the next reader knows whether the
   workaround is needed in their context. Otherwise it cargo-cults
   into places it doesn't belong.

---

## Addendum 11: verifying espr is the same OMP blocker (2026-05-16)

After Addendum 10 reframed egemv's gap as OMP-specific, re-tested espr
to check whether the pointer-walk fix in Addendum 8 was also working
around the same OMP issue rather than a general 2-pointer-lockstep
codegen limitation.

Standalone test with `gcc -O3`, natural counter form `for (int i = 0;
i <= j; ++i) ap[kk + i] += x[i] * tmp`:

| compile             | inner-loop insts/iter | shape                |
|---------------------|----------------------:|----------------------|
| `gcc -O3 -c`        | 8                     | shared-index, single `add` |
| `gcc -O3 -fopenmp` (inside `#pragma omp parallel for`) | 10 | counter + 2 ptr adds |

The "10 insts" form has three `add`s per iter:
- `add $0x1,%esi`  (counter)
- `add $0x10,%rcx` (ap pointer)
- `add $0x10,%rdi` (x pointer)

Without OMP, gcc would have folded all three into one shared-index walk.
Same exact blocker as egemv (Addendum 10).

The pointer-walk fix in Addendum 8 reached parity by a different
mechanism: it removed the counter (saving 1 insn) and let gcc compare
two pointers instead. That's 9 insts/iter vs the natural form's 10
under OMP — enough to hit parity with gfortran's 9-insn migrated loop,
but not the same root-cause fix.

**Equivalent fix would have been the `char*` byte-offset walk from
Addendum 7.** Not retrofit-tested because the pointer-walk already at
parity (0.97–1.09× vs migrated), and changing the source twice for
the same outcome is not worth it.

### The mistake pattern

This is the third (egemv, erot, espr) case where I diagnosed an
overlay gap as "gcc emits worse code than gfortran for this loop
shape" without first compiling the inner loop **standalone** to verify.
In all three the standalone gcc was fine; the gap was in the
*surrounding context*:

| overlay | blocker (initial guess vs actual) |
|---------|------------------------------------|
| egemv   | "gcc can't shared-index from natural C" → OMP outlining |
| erot    | "gcc keeps 2 pointers from natural C" → pointer-walk source form |
| espr    | "counter loop is worse than pointer-walk" → OMP outlining (same as egemv) |

### Rule

9. **When an overlay benches slower than migrated, compile the inner
   loop standalone before designing a workaround.** A standalone test
   distinguishes "gcc-vs-gfortran codegen gap" (rare) from
   "surrounding-context optimizer blocker" (common). The candidate
   blockers in this codebase:
   - `#pragma omp parallel` / `parallel for` outlining
   - pointer-walk source form (`for (; xp < xe; ++xp, ++yp)`)
   - `static inline` helper with a loop-invariant branch in its body
   - `restrict` placement that limits the alias-analysis loop
     transforms expect

   Skipping the standalone test costs honest framing in the commit
   message and in the findings doc. The patch may still land at
   parity, but for the wrong stated reason — and the next reader
   inherits a misleading explanation.

---

## Addendum 12: why OMP outlining blocks shared-index, and the cleaner fix (2026-05-16)

### Why

`restrict` is a function-parameter qualifier. When gcc's `-fopenmp`
outlines an `#pragma omp parallel` block, the captured pointers (`y`,
`a0`, `a1`, etc.) go into a struct passed by pointer to the outlined
function `func._omp_fn.0`. The outlined function loads pointers from
struct fields:

```c
void func._omp_fn.0(struct .omp_data_s *data) {
    T *y  = data->y;    // no restrict
    T *a0 = data->a0;   // no restrict
    T *a1 = data->a1;   // no restrict
    for (...) { ... }
}
```

The qualifier doesn't survive the trip through the struct. gcc's
shared-index transform (induction-variable folding across multiple
base pointers) needs to prove the bases don't overlap within the loop
body. With `restrict` on function args, the proof is trivial; with
struct-loaded pointers, gcc has to be conservative — three independent
induction variables, three `add`s per iter.

### Three workarounds, ranked

Tested on the egemv-style 3-array AXPY inner inside an OMP region:

| variant                                       | shared-index? | mechanism                                  |
|-----------------------------------------------|---------------|--------------------------------------------|
| natural C inside `_omp_fn`                    | ✗             | restrict lost through struct               |
| local `T *restrict y2 = y; ...` inside region | ✗             | gcc still tracks back to the struct field  |
| `char*` byte-offset walk                      | ✓             | bypasses IV-folding with one explicit IV   |
| `__attribute__((noinline))` helper            | ✓             | helper has its own restrict args; alias OK |

The `noinline` helper is the cleanest. Cost is one function call per
outer iter (~20 cycles); for any realistic inner-loop length M it's
negligible (<1% at M ≥ 64, <0.05% at M ≥ 1024).

### egemv refactor

Replaced the byte-offset walk with:

```c
__attribute__((noinline))
static void egemv_n_axpy2(int span, T *restrict y,
                          const T *restrict a0, const T *restrict a1,
                          T t0, T t1)
{
    for (int i = 0; i < span; ++i)
        y[i] = (y[i] + t0 * a0[i]) + t1 * a1[i];
}
```

called from inside the `#pragma omp parallel` region. Same inner-loop
codegen as the `char*` version (11 insns/iter, shared-index). Same
bench results within noise. Fuzz still passes at full long double
precision.

### Rule

10. **Inside an OMP-outlined region, factor the inner loop into a
    `__attribute__((noinline))` helper with `restrict` parameters.**
    The helper compiles in its own context where `restrict` is honored
    and gcc's loop optimizer fires normally. Call overhead is trivial
    (~20 cycles) compared to anything but pathologically short inner
    loops (M < 16). Prefer this over the `char*` byte-offset trick:
    same codegen, normal-looking C.

---

## Addendum 13: what actually blocks gcc's shared-index inside OMP (2026-05-16)

After Addendum 12 framed the blocker as "OMP outlining loses
`restrict` through the struct," tested every restrict variant and
alternative aliasing assertion to see if there was a way out short of
the noinline-helper or `char*` workarounds. **None of the obvious
fixes worked**, and successive minimal tests kept invalidating each
new hypothesis about the root cause.

### Variants tested

All against a 3-array AXPY (`y[i] = (y[i] + t0*a0[i]) + t1*a1[i]`)
inside `#pragma omp parallel`. Each row records whether gcc emitted
shared-index addressing (one `add`/iter, 11 insns) vs separate
pointer increments (two `add`s/iter, 13 insns).

| variant                                                | shared-index? |
|--------------------------------------------------------|---------------|
| `restrict` on function params, natural C loop          | ✗ (with `firstprivate`)            |
| `__restrict__` keyword on locals                       | ✗             |
| `restrict` on struct member (gcc extension)            | ✗             |
| `#pragma GCC ivdep` on inner loop                      | ✗             |
| `#pragma omp simd` on inner loop                       | depends — see below |
| `#pragma omp parallel for simd` on outer loop          | mixed         |
| `#pragma omp declare simd` on helper function          | ✓ — but only because gcc cloned the helper |
| `char*` byte-offset walk on inner loop                 | ✓             |
| `__attribute__((noinline))` helper                     | ✓             |

### Successively-falsified hypotheses

I went through three wrong root-cause hypotheses before landing on
something that survives the tests:

1. **"gcc can't shared-index from natural C"** — wrong. Compile the
   exact same inner loop in a standalone function (no OMP), and gcc
   emits shared-index without any hints.

2. **"OMP outlining loses `restrict` through the captured-vars
   struct"** — partly true (struct members really don't carry
   `restrict`), but not the actual blocker. Tested by re-establishing
   `restrict` on locals inside the parallel region; gcc still emitted
   two `add`s.

3. **"`firstprivate(i_lo, i_hi)` is the cause"** — also wrong. Tested
   by removing `firstprivate` and computing the per-thread slice
   directly from `omp_get_thread_num()` inside the region. Still two
   `add`s.

### The blocker, as far as I can isolate it

The pattern that survives all tests: **gcc loses IV-folding when the
inner loop's bounds depend on values its alias/IV analyzer can't
trace back to function parameters or compile-time constants**. Sources
of "opaque bounds" inside an OMP region include:

- `firstprivate` scalar captures
- Return values of `omp_get_thread_num()` / `omp_get_num_threads()`
- Anything derived from those (e.g. `i_lo = (M*tid)/nt`)

If the loop is `for (int i = 0; i < M; ++i)` where `M` is the outer
function's parameter, gcc applies IV-fold even inside the parallel
region. As soon as bounds come from a slice computation, it doesn't.

Pre-shifting pointers (`yp = y + i_lo`) and normalizing to
`for (i = 0; i < span; ++i)` doesn't help — `span` still traces back
to the opaque source.

### Why noinline helper works

The helper has its own function scope. Its `span` parameter is just
an `int` — no captured-source provenance, no struct-load history.
gcc's loop optimizer treats it as it would any function parameter and
applies IV-fold normally. The whole OMP outlining mess stays on the
caller side of the function boundary; the inner-loop codegen lives in
a clean scope.

```c
__attribute__((noinline))
static void inner(int span, T *restrict y,
                  const T *restrict a0, const T *restrict a1,
                  T t0, T t1) {
    for (int i = 0; i < span; ++i)
        y[i] = (y[i] + t0 * a0[i]) + t1 * a1[i];
}
```

### `omp simd` is interesting but unreliable

A bare `#pragma omp simd` on the inner loop *does* get gcc to emit
shared-index — as long as the loop bounds don't come from a
firstprivate or omp-runtime call. With the realistic per-thread slice
in scope, `omp simd` had no effect. So it works in toy examples and
fails in real ones, which is worse than just always using the
noinline helper.

### Rule

11. **Don't trust "obvious" restrict fixes inside OMP regions.** The
    cascade of failed hypotheses above documents specifically:
    `restrict` on locals, `__restrict__` keyword, struct-member
    restrict, `#pragma GCC ivdep`, and `#pragma omp simd` all *look*
    like they should propagate the alias guarantee through OMP
    outlining. In gcc 15 with `-O3 -fopenmp`, none of them do when the
    inner-loop bounds depend on per-thread slice computation. The
    workarounds that survive testing are (a) `char*` byte-offset walk
    and (b) `__attribute__((noinline))` helper with `restrict`
    parameters. Prefer (b): normal C, same codegen, no cast trickery.

---

## Addendum 14: the Fortran bench's "residual gap" is a measurement artifact (2026-05-16)

After restoring the char* shared-index walk for egemv N (Addendum 7
form), the Fortran bench still reported overlay at 0.91× of migrated
at OMP=1. Spent a session trying to explain the residual as a
scheduling / port-pressure / outer-loop-shape gap. **It was none of
those — it was a measurement artifact in the bench harness.** Two
distinct effects compounded:

### Effect 1: link layout / iTLB

The Fortran bench links `libeblas_parallel-gfortran-13.a` (the full
overlay archive) plus `libeblas_migrated.a`. Static-archive selection
pulls in `egemv.c.o` from overlay and its dependencies (the OMP
outlined helpers, `blas_omp_max_threads`, libgomp stubs, etc.), but
*also* pulls in everything those reference. The result: overlay's
egemv ends up ~123 KB from migrated's egemv in the linked binary,
spread across many code pages.

In the steady-state bench loop:
- Each overlay call jumps into `egemv_`, which calls a libgomp helper,
  then the outlined `egemv_._omp_fn.0`, then back. Several code
  pages touched per call.
- Migrated is a single Fortran `.o` — 1–2 code pages.

The iTLB (~64–128 entries) holds page translations for instruction
fetches. With both functions in a small standalone binary (within
~2 KB), every needed page stays resident. With overlay spread across
distant pages, the iTLB churns on each invocation — a page-walk costs
~10–50 cycles, and a few of them per inner-loop pass compounds to a
steady 5–10% throughput hit on the further function. Looks
indistinguishable from "this kernel is slower."

Confirmed by toggling `-Wl,--gc-sections` on the standalone C perf
harness: with the flag, the linker drops unused parallel BLAS
kernels, overlay's egemv code collapses to ~few KB, and the ratio
goes from 0.91× to 1.00× without touching any kernel code.

### Effect 2: gfortran's allocatable-LHS reallocation

Independent of layout, the Fortran bench's per-iter `Y = Y_init`
expands (under default flags, despite `-std=legacy`) into:
1. `realloc(Y, size)` — no-op when size unchanged, but still enters
   the glibc allocator state machine
2. Element-by-element scalar x87 copy loop (`fldt`/`fstpt`)

Both ov and mig calls do this before their timed section, but the
combination of `realloc` overhead + cache state asymmetry between
adjacent ov/mig calls penalizes the *second* call inconsistently.
This adds noise on top of effect 1; isolating it would need a
gfortran flag sweep that wasn't worth doing once we had the C
harness.

### How to get honest numbers

`tests/blas_parallel/perf/target_<name>/perf_*.c` — C harness with:
- Direct extern calls (no Fortran interface block)
- Aligned `aligned_alloc(64)` for A, X, Y
- Block timing (one clock pair per N calls — no per-call timer noise)
- One Y reset per timed block (avoids x87-slow value drift across
  many compounding GEMVs)
- CMake adds `-ffunction-sections -Wl,--gc-sections` per executable

With this harness, post-fix egemv reports parity with migrated:
N path 0.92–1.03× across N ∈ {128, 256, 512, 1024, 2048}; T path
1.04–1.45× (overlay ahead).

### Rule

12. **Bench numbers under 1.0× need verification before chasing.**
    The Fortran bench can manufacture 5–10% phantom gaps from binary
    layout alone. Before assuming a kernel is slow:
    (a) re-measure with the C perf harness (+ `--gc-sections`), and
    (b) if there's still a gap, compare inner-loop disassembly
    against migrated. A "residual sub-parity" without an inner-loop
    asm difference is almost certainly harness overhead, not kernel
    codegen.

---

## Addendum 15: full-overlay C-harness sweep (2026-05-17)

Generalized the kernel-isolated harness from §egemv (Addendum 14) to
every overlay routine and ran the full matrix at OMP=1, P-core pinned.
Generator at `scripts/gen_perf_harnesses.py` emits one
`perf_<name>.{c,cpp}` per (target, routine) — 195 harnesses across
kind10 / kind16 / multifloats. Sweep driver
(`scripts/run_perf_sweep.sh`) runs them all with per-routine `timeout`
so a single crash (or kind16 L3 hitting the 5-min cap) doesn't kill
the rest. Aggregator at `scripts/aggregate_perf_sweep.py` emits the
per-routine summary; `scripts/compare_perf_vs_fortran.py` cross-
references against the prior Fortran bench numbers under
`reports/{full-omp1,l2,l3-other,gemm-only}`.

Aggregate results in `reports/perf_sweep.{tsv,json,md}`;
divergence vs Fortran bench in `reports/perf_vs_fortran.md`.

### What the comparison shows: Addendum 14 generalizes

The most striking effect across the whole overlay: routines that the
Fortran bench reported as sub-parity collapse to parity under the C
harness. Top divergent cells (|Δ ratio| > 0.20):

| routine | key | size | Fortran | C harness | Δ |
|---------|-----|-----:|--------:|----------:|---:|
| egemm   | TT  | 256  | 1.760   | 2.689     | +0.929 |
| egemm   | TT  | 128  | 1.090   | 1.955     | +0.865 |
| egemm   | TT  | 512  | 3.560   | 2.902     | −0.658 |
| ygemm   | TN  | 64   | 0.600   | 0.989     | +0.389 |
| ygemm   | TT  | 128  | 0.620   | 1.005     | +0.385 |
| ygemm   | TN  | 128  | 0.590   | 0.968     | +0.378 |
| ygemm   | CN  | 64   | 0.650   | 0.996     | +0.346 |
| ygemm   | NN  | 64   | 0.730   | 1.062     | +0.332 |
| yhemv   | L   | 256  | 1.380   | 1.131     | −0.249 |
| yhemv   | L   | 512  | 1.390   | 1.141     | −0.249 |

All ~16 ygemm sub-parity cells the Fortran bench reported (0.59–0.85×)
flip to ~1.0× on the C harness. The pattern is consistent: every
routine the prior survey flagged as sub-parity for the overlay, where
the inner-loop disassembly is byte-identical to migrated, was the
iTLB / link-layout artifact — not codegen.

### One real codegen gap found: ygemmtr `!trans_a` path

ygemmtr (kind10 complex triangular-result GEMM) sat at median 0.806×
across all 24 (uplo × {NN,TN,NT} × size) cells in the sweep — the
only routine with a real gap after the C-harness filter.

Disasm showed the slow path was the `!trans_a` branch in the inner
loop:

```c
for (int l = 0; l < K; ++l) {
    T bl = B_(l, j);          /* or B_(j,l), or ~B_(j,l) */
    if (bl != zero) {
        const T t = alpha * bl;
        const T *al = &A_(0, l);
        for (int i = is; i < ie; ++i) cj[i] += t * al[i];
    }
}
```

Single FMA chain per `i`, gated on a runtime `bl != zero` check. This
is exactly the pattern Addendum 1 §kind10 complex documented as the
ygemm pre-fix shape: ~10-cycle x87 fmul latency × one chain =
throughput-bound at half of what 2 independent chains would reach.
ygemm's NN/NT paths got the K-unroll-by-2 fix in that addendum;
ygemmtr was missed.

Applied the same fix — split the K loop on `!trans_b`, `trans_b
&& !conj_b`, and `trans_b && conj_b` branches (hoisting the conj
out of the hot loop, see Addendum 1 §7's note that the in-loop
conditional defeats scheduling), each K-unrolled by 2:

```c
int l = 0;
if (!trans_b) {
    for (; l + 1 < K; l += 2) {
        const T t0 = alpha * B_(l,     j);
        const T t1 = alpha * B_(l + 1, j);
        const T *al0 = &A_(0, l);
        const T *al1 = &A_(0, l + 1);
        for (int i = is; i < ie; ++i)
            cj[i] += t0 * al0[i] + t1 * al1[i];
    }
} else if (!conj_b) {
    /* ...same, accessing B_(j, l) / B_(j, l+1)... */
} else {
    /* ...same with ~B_(j, l) / ~B_(j, l+1)... */
}
/* scalar tail for odd K */
```

Result: median 0.806× → **0.984×**, worst cell 0.748× → 0.910×.
Fuzz still bit-exact (200/200 pass, max rel err 5.96e-19).

### mswap: re-confirmation of Addendum 6's L1 noise rule

The sweep also flagged multifloats/mswap at median 0.923× — but at
20 iters and N=4096, the timed window is ~140 µs, below Addendum 6's
10-ms threshold for L1 routines. Re-running at 100 iters: 1.046× /
1.007× / 0.978× across N=4096/16384/65536. No gap.

The migrated mswap *does* have a gfortran-emitted unroll-by-3 on the
stride-1 path (visible as a `0xaaaaaaab` divide-by-3 magic constant
in the disasm); the overlay's auto-vectorized unroll-by-1 path looks
slower per-iteration on paper but reaches the same memory-bandwidth
asymptote at N≥16384. The original 0.754× cell was just sub-
threshold noise.

### Rules

13. **Build the per-routine C harness as a generator, not by hand.**
    195 routines × shapes makes hand-writing impractical, and the
    one bug in the dispatch (hardcoded `is_c=False` for symm,
    silently giving ysymm/xsymm/wsymm real-typed buffers half the
    needed size → heap corruption at runtime) is the kind that's
    obvious once you see all the harnesses next to each other and
    invisible when you're scanning one at a time. Lean on `extern "C"`
    + `BLAS_EXTERN` macro so the same generator emits valid C for
    kind10/kind16 and valid C++ for multifloats.

14. **The Addendum 1 fix list isn't a "done" list — it's a recipe.**
    Every place in the codebase with a single-FMA-chain rank-1 K loop
    on kind10 complex is a candidate for K-unroll-by-2. ygemmtr had
    the same shape and was missed. Look for: `for l in K` containing
    `cj[i] += alpha * b[...] * a[i+...]`. If you find one and ygemm
    has the K-unrolled form for the same orientation, port the same
    fix.

15. **20 iters at L1's smallest size is below the noise floor.**
    The C harness's default iters=20 is fine for L2/L3 but not for
    L1 routines at N=4096 (the wall-clock window is ~100 µs, timer
    resolution is ~µs, GC and context-switch jitter swamps small
    real effects). Either bump iters for L1 to ≥100 in the harness
    defaults, or accept that the L1 column at the smallest size is
    advisory only and require re-runs at higher iters before
    investigating any 0.85–0.95× cell. Same point as Addendum 6
    rule, applied to the new harness format.

---

## Addendum 16: `#pragma omp parallel for if(use_omp)` always outlines (2026-05-17)

espr (kind10 real symmetric packed rank-1) sat at median 0.92×
(worst 0.53× at U/128) for UPLO='U' while UPLO='L' was at parity.
The asymmetry made this a strong diagnostic — same routine, same
type, only the access pattern differs.

### What the disasm said (and didn't)

Comparing overlay's `espr_._omp_fn.{0,1}` against migrated's `espr_`:
- Inner loop: **byte-identical** between U and L on overlay, AND
  byte-identical to migrated (8 insns, single fldt→fmul→faddp→fstpt
  chain, walking two pointers via `add $0x10, %rdx` / `add $0x10, %rcx`).
- Outer overhead: U has slightly *less* per-outer-j work than L
  (the `apk = &ap[j*(j+1)/2]` address compute is one imul+shr+shl;
  L does similar plus a sub for the column-base offset).

Per the codegen rule from Addendum 2 (inner loops identical → no
codegen-level fix possible), nothing in the inner loop could explain
the gap. The gap had to be in the *caller's* per-call cost.

### What was actually wrong

`#pragma omp parallel for if(use_omp)` does **not** generate two
code paths (parallel vs serial). It always outlines the loop body
into a separate `._omp_fn.N` function and calls
`GOMP_parallel(_omp_fn, &data, 1, 0)` at runtime. The `if(use_omp)`
clause only affects whether the runtime actually forks threads —
the call into GOMP_parallel, the data-struct setup, the dispatch
into the outlined function, the `omp_get_num_threads()` /
`omp_get_thread_num()` queries at the start of `_omp_fn` — all of
that happens regardless.

At OMP_NUM_THREADS=1 this overhead is ~1–3 µs per espr_ call
(libgomp's GOMP_parallel + struct pack + outlined-fn prologue +
two libgomp queries). For an L1/L2 kernel whose pure inner work
is also a few µs, this is a measurable fraction.

espr's L path has more work per outer-j (longer inner loops at
small j) so the fixed dispatch cost amortizes. U's outer-j work
ramps from tiny (j=0: 1 element) upward, so the fixed cost is a
bigger fraction. Same overhead, different visible ratio.

### Fix: branch on `use_omp` in C source, two parallel loop bodies

```c
#ifdef _OPENMP
const int use_omp = (N >= ESPR_OMP_MIN && blas_omp_max_threads() > 1);
#else
const int use_omp = 0;
#endif

if (use_omp) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) { /* ...same body... */ }
} else {
    for (int j = 0; j < N; ++j) { /* ...same body, no pragma... */ }
}
```

The plain `for` in the `else` branch isn't outlined. No GOMP_parallel
call, no `_omp_fn` dispatch. The two bodies do duplicate source code
— acceptable cost for the saved dispatch overhead at OMP=1.

Result for espr:
- U @ 128:  0.529× → 0.927× (cold benchmark)
- U @ 256:  0.856× → 0.913×
- U @ 1024: 0.913× → 0.923×
- L:        unchanged (parity)
- median:   0.920× → 0.975×
- fuzz:     200/200 pass, bit-identical (max rel err 0.0)

Residual ~7% gap on U is sub-call-overhead noise; both halves now
sit at the same x87 codegen floor.

### Why this is bigger than just espr

Every parallel-BLAS overlay in this codebase uses the same
`#pragma omp parallel for if(use_omp)` idiom. At OMP=1 (the common
case for our perf sweep), every one of them is paying the same
~1–3 µs/call dispatch overhead. The cost is invisible for L3 and
large-N L2 kernels (per-call work is hundreds of µs), but for L1
and small-N L2 it's a 5–15% fraction. Several of the "documented
x87 floor" routines from Addendum 6 (yscal 0.93×, yescal 0.94×,
ydotu 0.93×, etc.) may have a chunk of their gap from this same
source — worth re-checking before chalking them up to codegen
ceilings.

The fix is mechanical: branch `use_omp` in C source. Not done
across the codebase in this round — flagged as a sweep candidate.

### Rules

16. **`#pragma omp parallel for if(use_omp)` is not "OMP only when
    needed."** It outlines unconditionally; the `if` clause only
    skips the actual thread fork. Caller pays GOMP_parallel +
    omp_get_* overhead per call regardless. At OMP=1 with a fast
    inner kernel, this is a measurable fraction of the call.
    When the per-routine ratio sits below 1.0× and the inner-loop
    disasm matches migrated (the Addendum 2 "no codegen gap"
    condition), check the omp pragma idiom before chasing more
    exotic explanations. Fix is a two-body branch on `use_omp` in C
    source.

17. **Asymmetric ratios across (uplo, side, trans) at the same
    routine are diagnostic.** If a routine has e.g. UPLO=U at
    0.6× and UPLO=L at 1.0× with identical inner-loop disasm and
    no algorithmic difference, the gap is in the outer / framing /
    dispatch — not the kernel. The denominator (migrated rate)
    being roughly constant across the two while the numerator
    (overlay) drops with smaller per-outer work is a tell that
    fixed per-call overhead is the cause.

---

## Addendum 17: closing every sub-parity routine — there is no x87 floor (2026-05-17)

The C-harness sweep at OMP=1 flagged ~8 routines as sub-parity (median
0.85–0.95×). Earlier addenda (1, 2, 6) had characterized these as
"the documented x87 stack ceiling" or "the codegen floor" — the
implication being that gcc's x87 backend couldn't be made to match
gfortran's emission, and the gap was a fundamental ceiling. **That
characterization was wrong on every single instance.** Each
sub-parity routine turned out to have a specific, fixable cause; none
were genuine floors. Working through them in order:

### 1. ygemmtr — missing K-unroll-by-2

`ygemmtr` (kind10 complex triangular-result GEMM) sat at median
0.806×. Inner `!trans_a` loop was `cj[i] += (alpha*bl) * al[i]`
across K — single FMA chain per i, bottlenecked by ~10-cycle x87
fmul latency. **This is exactly the shape Addendum 1 §kind10
complex documented for ygemm's NN/NT paths, and exactly the same
fix (K-unroll-by-2 with two independent FMA chains, conj hoisted
out of the loop) applied.** Median 0.806× → 0.984×. The Addendum 1
fix list was a recipe, not a done list (rule 14).

### 2-5. espr, esyr, egbmv, ygemv — OMP outline overhead at OMP=1

Four kind10 routines using `#pragma omp parallel for if(use_omp)`
showed asymmetric sub-parity at small N. Addendum 16 documented the
root cause: that pragma outlines the loop body into a `._omp_fn`
function unconditionally, and the caller pays GOMP_parallel + libgomp
query overhead per call even when `use_omp` evaluates to false at
runtime. Fix: branch `use_omp` in C source with two parallel loop
bodies (one with the pragma, one without).

  - `espr` U: 0.92× → 0.98× (initial), then 0.98 → 1.00× after
    char* shared-index walk for the U non-OMP path inner loop
  - `esyr`: 0.985 → 1.00
  - `egbmv` T: 0.87 → 1.00
  - `ygemv` T/C paths: 0.83–1.00 → 1.00–1.07

### 6-7. ydotc, ydotu — 2-acc unroll overflowing x87 stack

Addendum 1 §kind10 complex had warned: "each _Complex long double
multiply needs ~6 fp80 slots; 2 accs + temp slots overflow the
8-deep x87 stack and force fxch/spill." The 2-acc unroll added in
commit add00f58 was meant to expose ILP, but the actual measurement
contradicted the prediction — overlay's 2-acc inner loop has 16
insns/element vs migrated's 1-acc 14 insns. Reverted to single
accumulator with pointer-walk. Median ydotc 0.868 → 0.992; ydotu
0.931 → 1.000. **The 2-acc was actively hurting.**

### 8-10. yscal, yescal, erotm — gcc emits products in source order

The most surprising group. yscal complex inner loop had 15 insns +
2 fxch per element while migrated had 14 + 1 fxch — gcc emits one
extra `fxch` after the first store, in the imag-part computation.
Tried `restrict`, scheduler flags, `-fcx-fortran-rules` (already
on) — none changed it.

The actual cause: gcc emits the products of an `a*b + c*d` form
**in source order**. For complex multiplication, the imag-part
expansion is `xr*ai + xi*ar`. After the first store consumes
some x87 stack, the value left on top is `xi`, not `xr`. So gcc's
attempt to compute `xr*ai` first needs an `fxch` to bring `xr`
to the top, then another `fxch` to set up the second multiply.
gfortran (or gcc when expression-tree reordering kicks in) picks
the order `xi*ar + xr*ai` — using the top-of-stack value first,
saving one `fxch`.

**Fix:** rewrite with explicit `__real__`/`__imag__` and the
top-of-stack-first product order:
```c
const long double xr = p[0], xi = p[1];
p[0] = xr * ar - xi * ai;
p[1] = xi * ar + xr * ai;     /* xi term first */
```
Inner loop: 14 insns + 1 fxch — byte-equivalent to migrated.

Same trick applied to erotm's flag-unswitched paths. yescal had a
related pathology: `(__imag__ *p * alpha) * 1.0iL` triggered gcc's
full complex-multiplication expansion (4 fmul + 2 fadd, including
products by zero); rewriting as two independent real multiplies on
the (re, im) pair eliminated half the work.

  - `yscal`  0.931 → 0.999
  - `yescal` 0.942 → 1.022  (overlay now ahead)
  - `erotm`  0.946 → 1.003

### 11. espr U — gcc misses shared-index fold across mismatched bases

espr U direct path had inner loop:
```c
for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
```
gcc emits TWO pointer increments per iter (apk and xp). Migrated's
L path emits ONE — gfortran folds both into a single shared-index
walk because both pointers start from offsets in a shared base
register. gcc/gfortran neither fold this for the U path because
apk and xp start from different bases (apk from &ap[kk], xp from
&x[0]). The Addendum-7 char* byte-offset walk forces the fold —
explicit byte index `k`, derive both T* from `(char*) + k`. Inner
drops from 9 to 8 insns. Median 0.975 → 1.000.

### 12. mswap — sweep size grid was blind to the real signal

mswap (multifloats real DD swap) was flagged at median 0.923 in
the sweep. Re-running at higher iters at the same sizes showed
parity (1.0×). **But running at smaller N revealed the real
picture**: overlay 2.67× faster at N=64, 2.58× at N=128, 1.90× at
N=256, converging to parity by N=2048+. Migrated `mswap_` has
~30-50 cycles of per-call overhead (stack canary, 5 callee-saved
register pushes, 48-byte stack frame, magic-divisor count/3 setup
for the gfortran-emitted unroll-by-3 path) that dominates at
small N. Overlay's tight 6-insn SIMD-by-1 loop has no such
per-call cost.

This was the real lesson: **the sweep's L1 default size grid
{4096, 16384, 65536} is in the memory-bandwidth-saturated regime
where every L1 kernel reads the same number of cache lines per
second.** Both overlay and migrated cap at ~35 GB/s there, ratios
all collapse to ~1.0×, and the sweep flag of 0.923 was just sample
jitter. The interesting signal is at smaller N where per-call
overhead is the dominant cost — overlay's lean prologue/epilogue
(no stack canary, fewer callee-saved pushes) translates to 1.4–7×
wins.

### The extended L1 size grid

Updated `scripts/gen_perf_harnesses.py`: L1 default size grid changed
from `{4096, 16384, 65536}` to `{64, 128, 256, 512, 1024, 2048, 4096,
16384, 65536}`. Default iters bumped from 20 to 200, warmup from 3
to 20 (so the smallest-N cells get ≥10 ms timed windows). Re-bench
of all 57 L1 routines at the new grid shows:

  - kind10 copy/swap: 5–6× at N=64
  - kind10 function-return (asum/iamax/dot/nrm2): 1.1–2.5× at N=64
  - kind10 vector ops: ~1.0× at all N (parity)
  - kind16 (libquadmath): ~1.0× at all N (per-op libcall overhead
    dominates so call setup is invisible)
  - multifloats: 3–157× at N=64; 5–25× steady-state. The wins are
    from SIMD DD kernels vs gfortran's scalar DD ops.

### Bottom line

**Zero sub-parity routines remain in the entire 195-routine sweep**
after these fixes. The "x87 floor" framing from earlier addenda is
retracted; there is no architectural floor. Every gap had a specific
cause and a specific fix.

The original wrong intuition ("gcc's x87 backend is at its limit")
came from comparing insn counts and concluding the codegen was
already optimal. The actual situation was that gcc was emitting
1–2 extra instructions per iter from specific patterns (source-
order product emission, missed shared-index fold, OMP-outline
overhead, redundant complex-mul expansion of `* 1.0iL`) that small
C-source rewrites could close.

### Rules

18. **"Codegen floor" is the wrong default explanation.** Every
    sub-parity routine in this overlay turned out to have a
    specific fixable cause. The categories encountered were:
    missing reuse of an Addendum 1 recipe (K-unroll for kind10
    complex rank-1), OMP-outline overhead at OMP=1 (Addendum 16),
    over-unrolling that overflows x87 stack capacity, gcc's
    source-order product emission picking the wrong operand to
    leave on stack top, gcc's missed shared-index fold across
    mismatched base registers, and idioms (`* 1.0iL`) triggering
    expensive expansions. When the inner-loop disasm is identical
    and the ratio is still below parity, check those in order
    before assuming a hardware ceiling.

19. **L1 routines need a small-N column in the sweep.** Default
    L1 sizes of {4096, 16384, 65536} hide the interesting signal:
    those sizes are all memory-bandwidth-saturated, every kernel
    converges to the same throughput, and ratios collapse to ~1.0×
    regardless of code quality. **The actual L1 performance
    differentiator is per-call overhead**, which only shows up at
    N≤~1024. Default L1 size grid is now `{64, 128, 256, 512, 1024,
    2048, 4096, 16384, 65536}` with iters=200.

20. **Per-call overhead matters more than inner-loop unrolling at
    small N.** For mswap on multifloats DD: migrated has a tightly-
    unrolled-by-3 inner loop, but its function setup (stack canary,
    5 callee-saved regs, 48-byte stack frame, divide-by-3 count
    arithmetic) costs ~30-50 cycles per call. Overlay's plain
    SIMD loop with minimal prologue beats it 2.67× at N=64. The
    unroll buys nothing once memory bandwidth is the cap, and
    costs measurably at small N — a textbook "this optimization
    is universally good" that's actually wrong in the regime
    where it would supposedly matter.

## Addendum 18: etrsv LTN — inner walk must mirror outer descent (2026-05-18)

### Symptom

After the full sweep declared zero sub-parity, a follow-up re-run of
`perf_etrsv` (iters=50) surfaced a cliff that the original sweep grid
clipped: `kind10 etrsv LTN` at N=1024 ran at 0.43× of migrated, and
at N=2048 still 0.72×. Every other key (UNN/UTN/LNN at all sizes,
LTN at 512 and below) was at parity.

```
etrsv  LTN   512   50    2.0487    1.6187   1.266×   (overlay wins)
etrsv  LTN  1024   50    0.0266    0.0617   0.431×   ←  cliff
etrsv  LTN  2048   50    0.0161    0.0224   0.719×
```

The shape is unmistakable: in-cache (N=512) overlay wins comfortably;
out-of-cache (N≥1024) overlay collapses to roughly half of migrated.
Pure throughput drops for both (~30× from N=512 to N=1024 — the
expected L2→memory cliff for a 16 MB triangular matrix at 16-byte
elements), but overlay drops harder than migrated.

### Cause: inner-loop direction vs outer descent

The LTN branch solves `Aᵀx = b` with `A` stored lower. The reference
algorithm iterates the outer `j` *backward* (`j = N..1`), accumulating
`temp -= A(i,j)*x(i)` over the column-`j` slice `i = j+1..N`.

Migrated Fortran walks the inner `I` loop *backward* as well:
```fortran
DO 140 J = N,1,-1
    TEMP = X(J)
    DO 130 I = N,J + 1,-1            ! ← backward
        TEMP = TEMP - A(I,J)*X(I)
130 CONTINUE
    ...
140 CONTINUE
```

Overlay (pre-fix) walked the inner forward, because the natural C
idiom for a contiguous range is `for (k = i+1; k < N; ++k)`:
```c
for (int i = N - 1; i >= 0; --i) {
    T t = x[i];
    const T *ai = &A_(0, i);
    for (int k = i + 1; k < N; ++k) t -= ai[k] * x[k];   /* forward */
    ...
}
```

Functionally equivalent. Performance-wise, not equivalent under cache
pressure. At N=1024 each column of A is 16 KB (long double × 1024)
and the full matrix is 16 MB — well past L2 (1 MB) and the
last-level cache (12 MB on this part). x is 16 KB, fits in L1.

**Forward inner under descending outer**: at outer iter `i`, the
last memory address touched is `x[N-1]`. The very next outer iter
`i-1` opens with reading `x[i-1]` — at the *opposite end* of the
range. Under L2/L3 pressure from streaming column A, x is no longer
guaranteed to live in L1 (the column reads also push x out as victim
lines), and prefetcher state from the forward stream of x is no
help at the new start address. x effectively gets re-fetched every
outer iter.

**Backward inner under descending outer**: at outer iter `i`, the
last x address touched is `x[i+1]`. The next outer iter `i-1` opens
with reading `x[i-1]` — adjacent to the previous end. The backward
prefetch stream from the previous iter has already touched (or
prefetched) the next-needed addresses. x stays hot.

The fix is a one-line change to the inner loop:
```c
/* before */ for (int k = i + 1; k < N; ++k) t -= ai[k] * x[k];
/* after  */ for (int k = N - 1; k > i;  --k) t -= ai[k] * x[k];
```

Result (same harness, same iters):

```
                 before          after
etrsv  LTN  512  2.0487 vs 1.62  1.7202 vs 1.89   1.266× → 0.910×
etrsv  LTN 1024  0.0266 vs 0.062 0.0605 vs 0.063  0.431× → 0.962×
etrsv  LTN 2048  0.0161 vs 0.022 0.0227 vs 0.023  0.719× → 1.003×
```

Net trade: lose a small in-cache edge at N=512 (1.27× → 0.91×;
backward inner is slightly worse for the prefetcher when working
set fits in L1), gain everything at N≥1024 where the bug actually
hurts. Out-of-cache wins universally for a routine that exists to
be called on large matrices.

The other three keys (UNN/UTN/LNN) were already at parity at large N
because their outer/inner directions happen to align with their access
pattern naturally:
- UNN: outer backward + inner forward over `x[0..i-1]` — last addr
  touched is `x[i-1]`, next outer iter opens with `x[i-2]`. Forward
  walk is the right alignment here.
- UTN, LNN: outer forward + inner forward — last addr touched at
  the high end, next iter opens slightly past it. Natural prefetch.

Only LTN had the directional mismatch.

### Same bug in ytrsv

`ytrsv` (kind10 complex) has the same LTN/LCN shape and was hiding two
even worse cells in the original sweep:

```
                 before          after
ytrsv  LTN  512  0.291×          0.996×
ytrsv  LTN 1024  0.736×          0.992×
ytrsv  LCN  512  0.313×          1.001×
ytrsv  LCN 1024  0.724×          1.006×
```

Same one-line fix in both the `conj_a` and non-`conj_a` branches.
Sister routines `xtrsv`/`qtrsv`/`mtrsv`/`wtrsv` are compute-bound
(libquadmath or multifloats software ops dominate), so the cache
direction can't matter there — verified at parity in the sweep.

### Why the sweep missed it

The original sweep grid for L2 trsv routines stopped at N=1024 (which
caught the cliff but the 0.43× row sat among ~200 other rows and the
sub-parity-rollup script's threshold treated it as borderline). The
50-iter re-run with N=2048 added makes the regression unambiguous —
it's not a noise outlier, it scales with N.

Lesson: re-aggregate sub-parity not just by *cell count* but by
*maximum gap per routine*. A single 0.43× cell hidden in a routine's
otherwise-parity row matters more than three routines stuck at 0.95×.

### Rule

21. **Inner-loop direction is part of the algorithm, not the
    style.** When the outer loop descends, walk the inner loop
    in the matching direction even when the C idiom prefers
    ascending. For triangular accumulators the directional choice
    decides whether x stays hot in L1 across outer iters, and
    it only matters once the matrix work spills L2 — i.e. at
    exactly the size you ship the routine to handle. Match the
    Fortran reference's loop direction unless you have a specific
    reason to deviate.

    **Refinement (2026-05-18, ex loop-direction survey):** Rule 21
    applies to **compute-light targets where memory bandwidth catches
    up to compute (kind10/x87)**. For compute-bound targets
    (kind16/libquadmath, multifloats DD) the cache-direction cliff
    does not materialize — FP work dominates and x stays hot
    regardless — and Intel's hardware streaming prefetcher detects
    ascending strides more aggressively. There, ascending inner
    wins; keep forward regardless of the reference. Measured: kind10
    `etrsv LTN` 0.43→0.96×, `ytrsv LTN/LCN` 0.29→1.00× with
    backward inner; but `qtrsv LTN/x2` 1.106→1.002× regression at
    N=512 and similar for `xtrsv`, `mtrsv`. kind10 `*trsv`/`*trmv`
    converted; kind16 + multifloats deliberately kept forward.
    Full per-site survey and measurement table: see
    `doc/archive/loop-direction-survey-20260518.md`.

## Addendum 19: ytrsv U-T K-unroll asymmetry — conj path resists the trick (2026-05-18)

### Symptom

The full sweep flagged `ytrsv U-T-{N,U}` at N=256/512 as sub-parity
(0.82–0.90× of migrated). Direction was already aligned with the
Fortran reference (outer forward + inner forward — Addendum 18's
backward-inner fix does not apply: it's specifically for *descending*
outer loops). So the gap was a different bottleneck.

### Cause: single complex-fmul dep chain on the U-T accumulator

The U-T inner loop is `t -= ai[k] * x[k]` over `k = 0..i-1` with a
single complex accumulator `t`. Each iteration's `*` and `-=` are
gated on the prior iter's `t`. With `_Complex long double` on x86-64
(`-fcx-fortran-rules`), gcc expands the multiply to four scalar
fmuls on the x87 stack; the ~10-cycle fmul latency serializes the
chain at roughly 12 cycles/element vs migrated's ~11.

Same shape as Addendum 11's ygemm rank-1 fix — two-way K-unroll
splits the chain:

```c
for (; k + 1 < i; k += 2) {
    t0 -= ai[k]     * x[k];
    t1 -= ai[k + 1] * x[k + 1];
}
```

This recovered U-T from 0.82–0.90× to **~0.93×** consistently across
N=128/256/512.

### Asymmetry: same unroll on U-C **regresses** the routine

Surprising result. The conj variant has identical structure modulo a
`cconj()` (an fchs on the imag part):

```c
t -= cconj(ai[k]) * x[k];   /* U-C, single accumulator */
```

Applying the same K-unroll-by-2 here regresses U-C from a clean
~1.00× to **~0.91×** at all sizes. Verified across five runs each,
not variance.

Mechanism (inferred from disassembly): the extra `fchs` per multiply
displaces the x87 stack-slot scheduling. With one accumulator gcc
keeps `t` near the top across iterations; with two accumulators plus
two `fchs`-injected sequences it has to fxch more, and the resulting
schedule is worse than the un-unrolled baseline.

Combined attempts (scalar real/imag decomposition, scalar + K-unroll
with four accumulators tr0/ti0/tr1/ti1) all measured worse than
plain K-unroll on U-T and worse than un-unrolled on U-C.

### Fix

Apply K-unroll **only** to the non-conj branch:

```c
if (conj_a) {
    for (int i = 0; i < N; ++i) {
        T t = x[i];
        const T *ai = &A_(0, i);
        for (int k = 0; k < i; ++k) t -= cconj(ai[k]) * x[k];
        ...
    }
} else {
    for (int i = 0; i < N; ++i) {
        T t0 = x[i], t1 = ZERO;
        const T *ai = &A_(0, i);
        int k = 0;
        for (; k + 1 < i; k += 2) {
            t0 -= ai[k]     * x[k];
            t1 -= ai[k + 1] * x[k + 1];
        }
        if (k < i) t0 -= ai[k] * x[k];
        T t = t0 + t1;
        ...
    }
}
```

### Result

| ytrsv stat            | before | after |
|-----------------------|-------:|------:|
| min speedup           |  0.82× | 0.88× |
| median                |  1.00× | 1.00× |
| geomean               |  0.98× | 1.00× |
| sub-0.95× rows (of 36)|      6 |     5 |

Residual 7% on U-T vs migrated appears irreducible — gfortran builds
with strictly fewer flags (`-O3` only, no `-march=native`, no
`-ffp-contract=fast`) and is still faster, so it's not a flag gap.
The difference is gfortran's complex-multiply x87 instruction
scheduling, which gcc with `-fcx-fortran-rules` doesn't quite match.

### Rule

22. **K-unroll-by-2 helps single-accumulator dep chains for plain
    complex fmul on x87, but the conjugate variant resists the
    same transformation.** Apply K-unroll asymmetrically when the
    branch carries an inline conjugate — gate on `conj_a` and keep
    the conj path single-accumulator. Verify per-branch, don't
    assume the unroll transfers across `T`/`C` boundaries.

## Addendum 20: gfortran J-unrolls strided fallbacks too (2026-05-18)

### Symptom

A fresh `scripts/run_perf_sweep.sh` against current builds (the
committed `reports/perf_sweep.{tsv,md}` was generated against a
pre-Addendum-18 build and showed stale numbers) surfaced a systematic
N-path regression in the matrix-vector routines:

```
egemv N/x*     0.64x  across N=128..2048    (incx != 1, incy == 1)
egemv N/y*     0.62x  across N=128..2048    (incx == 1, incy != 1)
egemv N/x*/y*  0.51-0.65x                   (both strided)
ygemv N/x*     0.67x                        (same pattern, complex)
ygemv N/y*     0.83x
etrmv UNN      0.58x at N=1024              (triangle case)
```

40 perf cells in `egemv` alone, all sub-0.7×. Not bench variance —
the routines were systematically slow at every size.

### Root cause

Reading `egemv.c` against `egemv.f` (the Netlib reference, migrated
at `KIND=10`): the Fortran reference branches the N-path into THREE
paths:

```fortran
IF (INCY.EQ.1) THEN
    DO 60 J = 1,N
        TEMP = ALPHA*X(JX)
        DO 50 I = 1,M
            Y(I) = Y(I) + TEMP*A(I,J)        ! unit-stride y inner
50      CONTINUE
        JX = JX + INCX
60  CONTINUE
ELSE
    DO 80 J = 1,N
        TEMP = ALPHA*X(JX)
        IY = KY
        DO 70 I = 1,M
            Y(IY) = Y(IY) + TEMP*A(I,J)      ! strided y inner
            IY = IY + INCY
70      CONTINUE
        JX = JX + INCX
80  CONTINUE
END IF
```

Three branches: incx==incy==1, INCY==1 only, INCY!=1. My C code
collapsed branches 2 and 3 into a single "general stride" fallback
that used `iy += incy` unconditionally — preventing gcc from
recognizing the unit-stride y access pattern and J-unrolling.

But that's not the whole story. Even the strided-y branch (3) is fast
in Fortran. Inspecting the gfortran asm, the strided-y inner loop is
also J-unrolled by 2:

```
0x380: fldt 0x10(%rax)              ; load A(i,j+1)
0x387: add $0x10, %rax              ; advance A pointer by one long double
0x38b: fmul                         ; * temp1
0x38d: fldt -0x10(%rdx)             ; load y[iy]
0x390: faddp                        ; y[iy] + temp0*A(i,j) (accumulated earlier)
0x392: fldt 0x10(%rcx,%r10,1)       ; load A(i,j+1) for second column
0x397: fmul                         ; * temp2
0x399: faddp                        ; + temp1*A(i,j+1)
0x39b: fstpt -0x10(%rdx)            ; write y[iy] back
0x39e: add %r11, %rdx               ; advance y pointer by incy*sizeof(long double)
0x3a1: cmp %rdi, %rax
0x3a4: jne 0x380
```

11 instructions, 4 fmuls, processes ONE strided y[iy] update per
iter but reads from TWO columns of A. Each y[iy] load+store services
both column contributions, halving y memory traffic.

### Fix

Two changes per affected routine:

1. **Branch B**: when `incy == 1 && incx != 1`, route through the
   unit-y fast path with a strided x[jx] read. Same J-unroll-by-2
   shared-index walk as the (incx==incy==1) case.

2. **Strided-y J-unroll**: when `incy != 1`, manually replicate
   gfortran's auto-J-unroll-by-2 — process two columns of A per
   y[iy] write, halving y traffic on the strided path.

Applied to: egemv (commits 9a21723d, d6534703), ygemv (cfb01400),
etrmv UNN (9bcf52bb, same pattern but on the x[i] AXPY-style write).

### Result

```
egemv N/x*      0.64x  ->  0.95-1.05x   (all sizes >= 256)
egemv N/y*      0.62x  ->  0.95-1.05x
egemv N/x*/y*   0.55x  ->  0.95-1.05x
ygemv N/x*      0.67x  ->  0.85-0.95x
ygemv N/y*      0.83x  ->  1.00x
etrmv UNN @1024 0.58x  ->  1.00-1.60x   (often beats migrated)
```

About 80 perf cells closed across four kind10 routines.

### Why this kept hiding

The committed `reports/perf_sweep.tsv` was generated against a
pre-Addendum-18 build and showed even worse numbers (LTN@512 at
0.291x). After Addendum 18 landed, no one re-ran the sweep — so the
report stayed pinned to the pre-fix state. Today's regen against
current builds exposed both the closed LTN cliff and these strided
N-path gaps.

Lesson: `perf_sweep.{tsv,md}` is an artifact frozen at commit time.
Always regenerate before drawing conclusions from it. The `bench`
suite (Fortran-driven, ctest) hides these strided cases because it
only exercises `incx=incy=1`; you need `perf_<routine>` (C harness)
to see them.

### Rules

23. **Mirror the Fortran reference's branch structure for strided
    fallbacks, not just the algorithm.** If the reference has three
    branches (unit-unit, unit-y, general), implement all three. The
    middle branch is where strided-x + unit-y cells live, and they
    only get the fast unit-y inner if you split them out.

24. **J-unroll-by-2 also applies to strided-y inner loops.** Even
    when y is written with a variable stride, fusing two column
    updates per write halves y memory traffic. gfortran does this
    auto; gcc with `-fcx-fortran-rules -ffp-contract=fast` does not
    on the matrix-vector pattern. Hand-roll the unroll for any
    routine whose inner is `y[iy] = y[iy] + temp*A(i,j); iy += incy`.

25. **`perf_sweep.md` reflects a frozen build, not current state.**
    Treat the committed sweep as historical. Before claiming a cell
    is sub-parity, regenerate against the current build with
    `scripts/run_perf_sweep.sh`. The bench-side ctest output is a
    poor substitute: it only tests `incx=incy=1`.

## Addendum 21: etrsv TRANS='T' — same single-acc dep chain as ytrsv U-T (2026-05-18)

### Symptom

After Addendum 18 closed the LTN cliff, the residual etrsv sub-parity
clustered on TRANS='T' (UT and LT) cells with strided x:

```
etrsv UTU/x-1 @512    0.93x
etrsv UTU @256        0.95x
etrsv LTU/x-1 @256    0.95x
etrsv UTN/x-1 @256    0.96x
```

All TR='T' branches; TR='N' AXPY-style updates were already at parity.

### Cause: identical to ytrsv U-T (Addendum 19), applied to real

etrsv's TRANS='T' inner is a dot-product accumulator:

```c
T t = x[i];
for (int k = 0; k < i; ++k) t -= ai[k] * x[k];
if (nounit) t /= ai[i];
x[i] = t;
```

Single accumulator `t` makes each iter wait on the prior iter's fsub.
At x87 fmul latency ~7 cyc + fsub ~3 cyc this serializes at ~10
cyc/element. etrmv already had the split-acc K-unroll on its T paths;
etrsv didn't.

### Fix

Same pattern as etrmv T and ytrsv U-T:

```c
for (int i = 0; i < N; ++i) {
    T t0 = x[i], t1 = zero;
    int k = 0;
    for (; k + 1 < i; k += 2) {
        t0 -= ai[k]     * x[k];
        t1 -= ai[k + 1] * x[k + 1];
    }
    if (k < i) t0 -= ai[k] * x[k];
    T t = t0 + t1;
    if (nounit) t /= ai[i];
    x[i] = t;
}
```

Mirror form (backward k) applied to the LT branch where the cache-
direction fix from Addendum 18 already runs `k = N-1 .. i+1`. Both
unit-stride and general-stride fallbacks updated.

### Why no `!conj_a` gate (cf. Rule 22)

etrsv is real-only — no conjugate path that could regress under the
K-unroll's added register pressure. The unconditional K-unroll is safe.

### Result (5-run avg, BLAS_PERF_ITERS=40)

| Cell             | Before | After  | Δratio |
|------------------|--------|--------|--------|
| UTU/x-1 @512     | 0.930x | 0.989x | +0.059 |
| UTU @256         | 0.949x | 0.998x | +0.049 |
| LTU/x-1 @256     | 0.951x | 0.993x | +0.042 |
| UTU/x-1 @128     | 0.956x | 0.977x | +0.021 |

Overlay throughput up 3–6% on TR='T' cells; small-N (≤128) cells move
within ±3% (noise floor). The big-win cells are at N≥256 where the
inner-loop count is large enough for the split-acc trampoline overhead
to amortize.

### Lesson

When a routine has both a "y += A·x" AXPY path AND a "y[j] = Σ A·x"
dot-product path, the dot path almost always needs K-unroll-by-2 (x87
fmul latency vs. single accumulator). Check etrmv T, etrsv T, and the
symv/hemm dot phases together — they share the same shape.




## Addendum 22: s*mv strided fallback — a ~10% gap I can't close (2026-05-19)

### Symptom

After Addendum 21's etrsv T fix, the residual kind10 sub-parity is
dominated by the *symmetric* MV triad in the strided (incx≠1 or
incy≠1) fallbacks:

```
esymv strided:  47 cells at 0.88-0.92x
esbmv strided:  47 cells at 0.88-0.91x
espmv strided:  46 cells at 0.87-0.93x   (140 total)
```

Unit-stride paths for the same routines are at parity (0.99-1.01×).
Migrated runs at ~1.85 GFs strided, overlay at ~1.65 GFs.

### What didn't work

Three structurally-plausible fixes were tried and reverted:

1. **Cumulative `ix/iy` pointer-walk** (mirroring Fortran `IX = KX;
   ix += incx` instead of `kx + i*incx` recomputation). No effect —
   gcc already folds the multiply into a cumulative add via SCEV.

2. **Factor `A(k,i)` into a local** so it loads once per iter:
   ```c
   const T aki = ai[k];
   y[ky + k*incy] += temp1 * aki;
   temp2 += aki * x[kx + k*incx];
   ```
   gcc emits `fld %st(0)` at the inner — *exactly the same insn
   sequence as migrated's gfortran-compiled inner* (15 insns: fldt A,
   fld dup, fmul, fldt y, faddp, fstpt y, advance y, fldt x, advance
   x, fmulp, faddp, cmp, jne). Verified via objdump. **Ratio
   unchanged.**

3. **K-unroll-by-2 with split dot accumulators** (`t0/t1 + t0+t1`).
   *Regressed* the ratio from 0.90 → 0.86: the doubled per-iter
   bookkeeping (extra `k+1` index calc, larger inner with 5+ x87
   stack slots) costs more than the latency savings.

### Hypothesis: gcc's outer-loop overhead

With identical inner asm, the gap must be in the *outer*. gfortran's
migrated emits ~10 insns of setup per outer iter (lea/sub/add to
compute A column base + i_lo bound). gcc emits ~15-20 insns, including
extra `mov` to stack slots for register-pressure relief. At N=256
inner-iter counts of ~128, the outer setup is ~5-10% of total cycles
— which roughly matches the observed gap.

This isn't fixable from C source: gcc's register allocator and CSE
choices for this pattern are locked. Vector intrinsics could win
back the gap but would break the kind10 long-double path.

### Lesson

For dot+AXPY combined inners with strided y, the 0.90× strided floor
is real and structurally hard. Don't spend more time on it without
a measurement budget for vector intrinsics / asm-level rewrites.
Tag the cluster and move on.

### Rules

26. **Confirm `fld %st(0)` is present before claiming an inner-loop
    fix.** When migrated has it and overlay doesn't (the gcc 2-fldt
    redundant-load pattern), `const T x_local = expr;` will usually
    make gcc emit the dup. When *both* already have `fld %st(0)` and
    the inner-loop insn count matches, the gap is in the outer — and
    likely uncloseable from C.

27. **K-unroll-by-2 is not universal.** It helps when the inner has a
    single-acc dot product chain on x87 (10+ cyc/element latency).
    It *hurts* when the inner already has two parallel chains (e.g.
    AXPY + dot combined): the doubled bookkeeping eats the latency
    savings and the extra accumulators spill x87 stack. Try on dot-
    only paths first; revert immediately if the strided cell ratios
    move <0.

## Addendum 23: complex K-unroll-by-4 spills x87 stack — keep at 2 (2026-05-19)

### Context

The 3-trial median sweep at OMP=1 left **ytrsv UTN/UTU @ N=256/512**
at 0.92-0.93× — a real structural gap, tight across 5 reruns, despite
already having K-unroll-by-2 on the non-conj branch (Addendum 19).
Tried K-unroll-by-4 to add two more parallel chains:

```c
for (; k + 3 < i; k += 4) {
    t0 -= ai[k]     * x[k];
    t1 -= ai[k + 1] * x[k + 1];
    t2 -= ai[k + 2] * x[k + 2];
    t3 -= ai[k + 3] * x[k + 3];
}
```

### Result

**Regressed** UTN/UTU @ N=256 from 0.93× → 0.75× (10 trials). Reverted.

### Why

`_Complex long double` is two `long double`s. Each accumulator occupies
two x87 register-stack slots. 4 accumulators × 2 = 8 slots — exactly
the entire x87 stack. With no slots left for `ai[k]` and `x[k]` operand
loads, gcc spills every load to memory and re-loads twice (once for
the real-part fmul, once for the imag). The 4-way split increases
parallel chains but the spill overhead more than wipes it out.

Real-valued long-double would have headroom (4 acc × 1 = 4 slots,
4 free for operands) — so the rule below is complex-specific.

### Lesson

Complex long-double is at the limit at K=2. The remaining 7% gap on
ytrsv U-T is inherent to x87 stack width plus dependent-chain latency
and is not closeable from C source. Document and stop.

### Rules

28. **For `_Complex long double` dot accumulators, max K-unroll is 2.**
    Each cmplx slot is 2 x87 stack registers; 4 accumulators = 8 slots
    = full stack with no room to load operands. Same code on real-long-
    double would have headroom. Don't try K-unroll-by-4 on complex
    paths even when "more parallelism would help": the spill cost is
    larger than the latency win.

## Addendum 24: K-unroll-by-2 regresses banded inners — short loops don't amortize (2026-05-19)

### Context

3-trial median sweep showed **ygbmv T 8 cells at 0.94-0.95** (TRANS='T',
non-conj). Same single-acc complex fmul dep-chain pattern as the
ytrsv U-T case from Addendum 19, so K-unroll-by-2 was expected to lift
ratios. Applied to both the stride-1 and strided fallback non-conj
branches.

### Result

**Regressed all T cells.** Sample re-bench:

```
ygbmv T 128       0.947 → 0.91   (regressed)
ygbmv T/x2 128    0.99  → 0.83   (regressed badly)
ygbmv T/y2 128    1.00  → 0.83   (regressed badly)
```

Reverted immediately.

### Why

Band MV inner-loop length is `min(KL+KU+1, M)` — for the perf bench's
defaults, that's roughly `M/8 + 1` (so ~17 at N=128, ~33 at N=256).
The bookkeeping for K-unroll-by-2 (extra index, second accumulator
init, tail handling, final `s = s0 + s1` reduction) is roughly fixed
~6 insns per outer iter; the latency-hiding benefit per inner iter
scales with inner length. With inner=17, the bookkeeping outweighs the
latency win and the unroll loses.

ytrsv U-T's inner is `i-1` long, growing from 0 to N-1 — average ~N/2,
so inner=64–256. That's enough for the K-unroll-by-2 win to pay off.

### Lesson

Inner-length matters as much as dep-chain structure. K-unroll-by-2 on
single-acc complex chains only wins when inner ≥ ~32 iters. For banded
ops (gbmv, sbmv, tbmv) the inner is bounded by K+1 — usually too short.
Don't transplant a fix from a full triangular/symmetric MV to a banded
version without re-benching at the actual band-width.

### Rules

29. **Bandwidth-bounded inners (gbmv, sbmv, tbmv) are typically too
    short for K-unroll wins.** The bench default band-width is M/8 ≈
    32-64 elements; subtract the bookkeeping insns and the unroll
    breaks even or loses. Test on the actual band width before
    committing. For dot+AXPY combined inners on banded ops, K-unroll
    is essentially never the right tool — Rule 27 already covers
    combined inners; this rule covers the short-inner case for pure
    dots.

### Addendum 24 follow-up: confirmed inner asm matches migrated, gap is structural (2026-05-19)

After Rule 30 / Addendum 25 (the 5+ trial, 20000-iter discipline),
ygbmv T 128 stays at median 0.94-0.96 — *tight* across 8 trials.
The cell is real, not bench-noise.

**Side-by-side inner-loop disassembly:**

migrated (gfortran, `ygbmv_+0xc00`, 19 insns):
```
fldt (%rax) / add $20 / fldt -0x10(%rax)   ; A.re, A.im
fldt -0x20(%rdx) / fldt -0x10(%rdx)         ; x.re, x.im
add %rsi,%rdx
fld %st(1) / fmul %st(4),%st                 ; dup A.im, * x.im
fld %st(1) / fmul %st(4),%st                 ; dup x.re, * A.re
fsubrp %st,%st(1) / faddp %st,%st(6)         ; cmul.re into TEMP.re
fxch %st(1)
fmulp %st,%st(2) / fmulp %st,%st(2)
faddp %st,%st(1) / faddp %st,%st(1)          ; cmul.im into TEMP.im
cmp %rax,%rdi / jne
```

overlay (gcc, `ygbmv_._omp_fn.0+0x100`, 18 insns — one fewer):
```
fldt (%rcx) / add $20 / add $20 / fldt -0x10(%rcx)
fldt -0x20(%rax) / fldt -0x10(%rax)
fld %st(3) / fmul %st(2),%st
fld %st(3) / fmul %st(2),%st
fsubrp %st,%st(1) / faddp %st,%st(5)
fmulp %st,%st(3) / fmulp %st,%st(1)          ; direct fmulp; no fxch
faddp %st,%st(1) / faddp %st,%st(2)
cmp %rax,%rdx / jne
```

Same algorithmic shape, same critical-path fpu chain (fmul → fsubrp →
faddp into spilled TEMP), overlay actually has one fewer instruction.

**Attempted fixes that didn't help:**

1. **K-unroll-by-2 (same column, split acc)** — regressed (Rule 29:
   banded inner too short to amortize bookkeeping).
2. **OMP outlining bypass** — moved the `use_omp` branch from `#pragma
   omp parallel for if(use_omp)` (always outlines, Addendum 16) into a
   C-level `if (use_omp) { ...pragma... } else { ...inline... }`.
   Confirmed GOMP_parallel call is gone from the use_omp=0 path. Bench
   unchanged (0.95 → 0.93, within noise). So outlining wasn't the
   bottleneck.
3. **2-j outer unroll (process columns j and j+1 together, sharing x
   reads, two independent acc s0/s1)** — regressed to 0.93 because of
   x87 stack spill: 2 acc × 2 slots + 2 x slots + 2 × 2 A slots = 10
   slots needed vs 8 available. Same Rule 28 problem as K=4 on
   complex.

**Conclusion:** the 0.94-0.96 floor is *not* in the inner loop, which
overlay wins; it's in the outer-loop dep chain (alpha * s reduction)
and/or gcc's worse register-allocation choices on the outer setup
(28 outer insns vs gfortran's 20). These are not fixable from C source
without breaking other paths or going SSE/AVX (which can't be done for
kind10 long-double in System V ABI).

Tag and stop. The cell stays at 0.94.

30. **3-trial median sweeps misclassify "at-parity with noise tail"
    cells as sub-parity.** A cell whose true distribution is ~0.97×
    typical with occasional 0.83× outliers (rare cache event, page
    walk, throttle) will show median 0.918 in a 3-trial sweep when
    2 of 3 happen to hit the outlier mode. 8-trial reruns recover
    the true median ~0.97. Before declaring a cell structurally
    sub-parity from a 3-trial sweep, rebench at ≥5 trials; only then
    classify as structural. The etbmv UNN strided "0.92 cluster" was
    a false-positive of this kind — at parity with a small outlier
    tail at incx≠1.

## Addendum 25: small-N benches are bimodal — need ≥20000 iters (2026-05-19)

### Symptom

A 5-trial sweep at BLAS_PERF_ITERS=2000 flagged **etbmv** with ~16
cells at median 0.70–0.95. Looking at the raw per-trial overlay GFlops:

```
etbmv UTN 128 (5 trials):
  overlay GFs:  0.86  1.21  1.18  1.25  0.87   ← bimodal!
  migrated GFs: 1.24  1.24  0.86  0.86  1.25   ← bimodal!
  ratio:        0.69  0.97  1.37  1.44  0.69
```

Both overlay and migrated show a **bimodal** GFlops distribution —
~0.86 (slow) or ~1.25 (fast) — and the ratio depends on whether they
happened to hit the same mode. The median ratio of 0.70 is not a code
artifact; it's a statistical artifact of bimodal noise.

### Root cause

At BLAS_PERF_ITERS=2000 with N=128, total wall-clock per call is
~10-20 ms. The "fast" mode is presumably hot-cache execution; the
"slow" mode is one-or-more cold accesses (TLB miss, page walk, or
cache line refetch) early in the iter loop that takes a fixed cost
amortized over only 2000 iters. At higher iter counts the fixed cost
amortizes away.

### Verification

Same etbmv UTN 128 at BLAS_PERF_ITERS=20000:

```
ratios across 5 trials: 0.912  0.962  0.978  0.992  0.924
                        → median 0.962, at parity
```

The bimodality vanishes; cell is at parity.

esymv L/x-1 128 at BLAS_PERF_ITERS=20000:

```
ratios across 5 trials: 0.893  0.904  0.905  0.922  0.931
                        → median 0.905, real structural floor
```

esymv is *genuinely* sub-parity (the Addendum 22 floor) at high iters.

### Implication

The whole exercise of 3-trial and 5-trial sweeps at 200-2000 iters
captured a mix of true structural sub-parity and bimodal-noise
artifacts. Reliable classification requires BLAS_PERF_ITERS=20000+
for small-N cells. Two cells fell out of the noise correctly:
- **etbsv UNN stride-1**: 0.84 at low iters, **0.96-0.99 at 20000** —
  the fix (forward inner walk) is real.
- **etpsv UTN stride-1**: 0.84 at low iters, **1.05 at 20000** —
  K-unroll-by-2 fix is real.

### Rules

31. **For sub-parity classification at small N (≤256), use
    BLAS_PERF_ITERS ≥ 20000.** Lower iter counts conflate true
    structural gaps with bimodal cache-warmup noise — both modes
    can have ~30% throughput delta, and ratios depend on whether
    overlay and migrated happen to coincide. The default
    BLAS_PERF_ITERS=200 from perf_sweep.sh is unsuitable for
    declaring structural sub-parity; even 2000 is borderline.

32. **A cell's "real" sub-parity needs *tight* trial spread** (max-
    min < 0.05 across 5+ trials) *and* low median (<0.95). A wide
    spread (e.g., 0.65–1.45) means the ratio is dominated by
    independent bimodal noise in overlay and migrated, not by any
    real code-level gap. Wide-spread cells should be rebenched at
    higher iters, not "fixed."

## Addendum 26: inner-OMP on ytrsv loses everywhere except UNN/N=512 (2026-05-20)

### Experiment

Built a Netlib-direct ytrsv variant (`ytrsv_netlib.c`) with
`#pragma omp parallel for` on the inner AXPY (TRANS='N') and on a
manual `re/im` reduction for the DOT branch (TRANS='T'/'C'). Built
two staged trees (current vs netlib), fuzzed netlib at OMP=1 and
OMP=4 against migrated (80 cases each, max err ~7e-19), then
benched all 80 cells (uplo × trans × diag × stride × N=64/128/256
+ N=512 for unit-stride) at high iter counts (20k–50k for small N,
8k for N=256, 2k for N=512), 3 trials, median across trials.

### Result

For unit-stride cells (where the OMP pragma fires):

```
key      N     cur-1   cur-4   net-1   net-4    mig
UTN     64     2.791   2.775   0.372   0.194    2.953
UTN    256     2.892   2.888   1.098   0.725    3.091
UNN    256     1.959   1.963   0.945   0.763    1.960
UNN    512     0.550   0.565   0.488   0.940 ←  0.555
LCN    128     2.659   2.682   0.648   0.378    2.676
```

Almost every unit-stride cell regresses 3-7× vs current at OMP=1
(even before OMP fork cost) — the manual `re/im` split-reduction
forces extra adds per iter and defeats the existing complex-fmul
K-unroll-by-2. OMP=4 is generally *worse* than OMP=1: fork-per-
outer-step cost (~10µs × N outer iters) dwarfs the inner FLOP work.

**One bright spot: UNN at N=512.** netlib-omp4 = 0.940 GFs vs
cur-1 = 0.550 vs migrated = 0.555. 1.7× speedup, beats migrated.
N=512 is large enough that the inner AXPY (~256 iters average) is
long enough to amortize the fork cost across 4 threads.

Strided cells (`x2`, `x-1`) are unchanged because the Netlib variant
keeps the strided fallback serial with no OMP pragma.

### Why this is the wrong tool

Triangular solve is sequential in the outer loop — each x[i]
depends on x[0..i-1]. The only naive parallelism handle is the
inner AXPY/DOT, but at the iteration count typical for sub-parity
cells (N=64-256), the inner length is 32-128 iters and the OMP
fork cost (5-10 µs per fork, ~3 forks per μs of FP work at small
N) overwhelms the win.

Real parallel TRSV needs a **block-recursive** structure: split A
into 2×2 blocks, recurse on the two diagonal blocks (smaller
TRSVs), and use a parallel GEMV for the off-diagonal trailing
update. The GEMV is where the OMP region pays off, because:
- One fork-join per recursion level (~log₂ N forks total) instead
  of N forks
- GEMV is fully parallel in M
- The recursive base case can use the existing single-thread kernel

This is ~200-300 LoC and a day of tuning. Not done in this
experiment.

### Rules

33. **Don't `#pragma omp parallel for` inside the outer of a
    sequential algorithm.** Fork-join cost is per-fork (5-10 µs);
    iterating N forks at small inner-length destroys any FLOP win.
    The pattern wins only when (a) outer is also parallelizable,
    or (b) the OMP region wraps the *entire* call (one fork) and
    threads pull work themselves.

34. **TRSV/TRMV cannot be naively row-parallelized.** Block-
    recursive is the only structural path: smaller TRSVs on the
    diagonal blocks + parallel GEMV for the trailing update.
    Inner-loop OMP is a dead end (Addendum 26).

## Addendum 27: kind16 xgemm 2D tile parallelism (2026-05-20)

### Context

xgemm's original OpenMP parallelism was a single `#pragma omp parallel
for` over its N dimension, gated at N ≥ 32. Used as a top-level call
that was usually fine, but the xtrsm_blocked trailing update has
shape `(mt, nrhs, jb)` — at small nrhs, only the N=nrhs axis (often
1-3 columns) carried parallelism, idling threads even though mt was
in the hundreds.

### Experiment

Refactor xgemm into a 2D collapse(2) over (M-tile, N-tile). Tile side
is adaptive: pick the largest power-of-two side s in [16, 128] such
that `s² ≤ M·N / (4·nthreads)` — targets ~4 tiles per thread for
load balance. `XGEMM_MB` / `XGEMM_NB` env knobs pin specific sides.

K stays serial inside each tile. Each (i, j) output element belongs
to exactly one tile under collapse(2), so the per-tile beta scaling
is race-free and bitwise-identical to the column-wise scheme.

### Result

xtrsm_blocked at M=1024, OMP=4, before vs after (cur_4/cur_1):

| nrhs | before | after |
|------|--------|-------|
| 1    | 0.99×  | 2.97× |
| 4    | 1.00×  | 2.51× |
| 8    | 1.03×  | 3.09× |
| 32   | 3.94×  | 3.91× |

Before, the trailing xgemm only parallelized when N=nrhs ≥ 32 → the
xtrsm_blocked path was stuck at serial-equivalent throughput for
nrhs < 32 (the "nrhs=32 cliff"). After 2D, parallelism scales at
any (M, N) ≥ 2 tiles.

Direct xgemm at OMP=4 (`mig/overlay` ratio, single-thread migrated):

| size | before | after |
|------|--------|-------|
| 64   | 1.0×   | 1.2-2.9× |
| 128  | 1.0×   | 1.5-2.3× |
| 256  | 3.0-3.8× | 3.0-3.8× |

No regression at size where the old N-only parallel was already
saturating; new wins at small-and-moderate sizes that were below the
old N ≥ 32 gate.

### Rules

35. **Parallelize the output, not one axis.** For GEMM-class kernels
    where the output is 2D (M × N), use `collapse(2)` over both axes
    rather than parallelizing one. The "good axis" is whichever is
    largest at runtime — and that's not knowable at compile time. 1D
    parallelism is fine for kernels with a structurally 1D output
    (gemv: y is a vector).

36. **Adaptive tile-side beats fixed.** `side ≈ sqrt(area /
    (4·nthreads))` clamped to a reasonable range gives ~4 tiles per
    thread regardless of (M, N, nthreads). Fixed tile sizes either
    under-saturate small problems or over-fragment large ones for
    the deployment thread count.

## Addendum 28: kind16 xtrsm — xtrsv-loop fast path + Amdahl-capped dispatch (2026-05-20)

### Context

At small nrhs, column-parallel xtrsm idles threads: with N=nrhs
columns and nt threads, scaling is `nrhs / ceil(nrhs/nt)`. For
nrhs=2 at nt=4, that's 2× speedup with 2 threads idle; for nrhs=3,
3× with 1 idle. xtrsm_blocked + 2D xgemm (Addendum 27) helps but
has overhead that wipes the win at very small nrhs.

The alternative for SIDE='L': xtrsm with nrhs columns is literally
nrhs independent xtrsv solves (mod the alpha pre-scale). Each xtrsv
can use *all* threads via xtrsv_blocked (parallel xgemv on the
trailing slice). For small nrhs, nrhs sequential xtrsv-blocked calls
beats column-parallel because each call fills the thread pool.

### Fast-path-vs-col-parallel crossover

Times in units of single-thread xtrsv time T, scaling factor s:

```
fast path = nrhs × T / s
col-par   = ceil(nrhs / nt) × T   (sawtooth in nrhs)
```

Crossover at `nrhs ≈ s`. Above that, col-par's "rounds" advantage
beats fast path's sequential xtrsv cost.

`s` is xtrsv_blocked's effective scaling, which is **Amdahl-limited
by the serial sub-solve fraction `nb / M`**:

- Total work ≈ M²/2 muladds.
- Per-block sub-solve = nb²/2 muladds → total sub-solve = M·nb/2.
- Serial fraction = nb / M → ceiling = M / nb.

| M    | nb=64 | serial frac | ceiling |
|------|-------|-------------|---------|
| 256  | 64    | 25%         | 4×      |
| 512  | 64    | 12.5%       | 8×      |
| 1024 | 64    | 6.25%       | 16×     |
| 2048 | 64    | 3.1%        | 32×     |

Effective scaling = `min(nthreads, M / nb)`. So the fast-path cap is
`min(nthreads-1, M/nb)`.

### Experiment

Add `xtrsm_xtrsv_loop_max(M)` that computes `min(env_override,
nthreads-1, M/nb_hint)`. The xtrsm_ entry routes SIDE='L', stride-1,
M ≥ 128, nrhs ≤ that cap into the fast path: pre-scale by alpha,
then loop xtrsv_ over columns.

Also lowered `XTRSM_OMP_MIN` (column-parallel gate) from 32 to 2 so
the col-parallel path picks up for nrhs ≥ 4 at any reasonable M.

### Result

Measured at M=1024, OMP=4, iters=25 (cur_4/cur_1):

| nrhs | path        | scaling |
|------|-------------|---------|
| 1    | fast        | 3.08×   |
| 2    | fast        | 3.19×   |
| 3    | fast        | 3.24×   |
| 4    | col-par     | 3.87×   |
| 5    | col-par     | 2.49×   (sawtooth) |
| 6    | col-par     | 2.96×   |
| 7    | col-par     | 2.72×   |
| 8    | col-par     | 3.87×   |
| 16   | col-par     | 3.85×   |
| 32   | col-par     | 3.85×   |

At nt=4, cap is min(3, M/64) = 3 — same value as the original fixed
default but adapts at higher nt. At hypothetical nt=64, M=1024 the
cap becomes 15 (= Amdahl ceiling - 1), preventing fast-path
over-extension into col-parallel's regime.

### Suboptimal sawtooth

nrhs ∈ {5, 6, 7} at nt=4 hit col-parallel's sawtooth (2.5-3.4×).
Fast path at those would give 3.1-3.3× — sometimes a small win, but
needs a per-nrhs dispatch (compare both paths' predicted time)
rather than a threshold. Deferred — the worst-case loss is bounded
and the dispatch complexity isn't worth a few percent at moderate
nrhs.

### Rules

37. **Derive dispatch thresholds from algorithmic Amdahl ceilings,
    not constants.** A "fixed threshold of 3" is correct only at the
    deployment's specific (nthreads, problem-size) combination.
    Compute it: scaling = min(nthreads, work_total / work_serial).
    The crossover for fan-out vs work-sharing dispatch sits at
    `nrhs ≈ scaling`.

38. **Column-parallel's effective scaling is `nrhs / ceil(nrhs/nt)`,
    not `min(nrhs, nt)`.** The ceiling-divide creates a sawtooth:
    at nrhs = nt, nt+1, 2·nt, … scaling is high; in between it's
    `nrhs / 2` at best. Fan-out (loop-of-parallel-xtrsv) can beat
    column-parallel in the sawtooth dips even when nrhs > nt.

## Addendum 29: single-parallel-region for blocked xtrs* + blas_omp cache anti-pattern (2026-05-20)

### The cache anti-pattern

`blas_omp_max_threads()` originally lived as a `static inline` body
in `common/blas_omp.h`:

```c
static inline int blas_omp_max_threads(void) {
    static int cached = 0;
    if (cached == 0) cached = omp_get_max_threads();
    return cached;
}
```

**Two bugs.**

1. **Per-TU caches.** `static inline` gives each translation unit
   its own `cached` variable. First call from xgemv.o locks
   xgemv.o's cache to whatever `omp_get_max_threads()` returned at
   that moment; xgemm.o caches independently. If they diverge, kernels
   silently disagree on whether to parallelize.

2. **Cache locked to first call.** Even centralized, the cache is
   set on the first `omp_get_max_threads()` reading and never
   re-checked. A test pattern that does:

   ```
   omp_set_num_threads(1); call_overlay();  // cache → 1
   omp_set_num_threads(4); call_overlay();  // still 1, no parallelism
   ```

   silently runs serial in the second call. Real production code
   typically sets `OMP_NUM_THREADS` once at startup, but perf
   harnesses (and any user code that switches thread counts) break.

Centralizing the cache to a single `blas_omp.c` fixed #1 but not #2.
Removed the cache entirely — `omp_get_max_threads()` is an ICV
lookup, well below the per-call BLAS overhead at any precision we
care about.

### Single-parallel-region pattern

The previous xtrsv_blocked / xtrsm_blocked shape:

```
for each diagonal block:           # serial outer
    sub_solve(small block)         # serial
    xgemv_(...) or xgemm_(...)     # opens its OWN omp parallel region
```

Per call: N/nb separate fork-joins. Each fork-join is ~5-10 µs on
libgomp; at N=1024, nb=64 that's 16 × 5-10 µs = 80-160 µs of pure
sync overhead. Also violates the "no OMP-using callee from inside
an OMP region" rule if the blocked variant is itself wrapped.

The refactor opens ONE `#pragma omp parallel` region around the
whole diagonal walk:

```
#pragma omp parallel
{
    tid, nt = ...
    for each diagonal block:           # serial walk INSIDE the region
        if tid==0: sub_solve()         # serial, others wait
        #pragma omp barrier
        partition_trailing(tid, nt)    # call *_serial_ on this slice
        #pragma omp barrier
}
```

Replaces N/nb fork-joins with **one fork-join + 2·N/nb barriers**.
Same order of sync cost but threads stay pinned across iterations —
better cache behavior on many-core nodes. Required `xgemm_serial_`
/ `xgemv_serial_` / `xtrsv_serial_` entries (same numerics, no OMP
pragmas) for the inner calls — the parallel entries (`xgemm_`,
`xgemv_`, `xtrsv_`) keep their own regions for direct top-level
use, and also pick up `omp_in_parallel()` guards so a stray call
from inside another region degrades to serial work-sharing rather
than nesting.

### Column-partition is naturally race-free for xtrsm_blocked

For xtrsm_blocked, partitioning by columns of B (each thread gets
[j_lo, j_hi)) is **race-free with zero barriers**:

- Pre-scale: thread T writes B[:, j_lo:j_hi]. Disjoint columns.
- Sub-solve at block ic: thread T reads/writes B[ic:ic+ib, j_lo:j_hi].
  Only its own columns.
- Trailing xgemm: thread T reads B[ic:ic+ib, j_lo:j_hi] (just wrote
  in sub-solve, same thread), writes B[i0:M, j_lo:j_hi].

No cross-thread sharing of B. A is read-only. So one fork-join per
xtrsm_blocked call, zero inner barriers. Each thread effectively
runs an independent xtrsm_blocked on its column slice.

This degenerates to column-parallel xtrsm at the algorithmic level —
the "blocked" structure adds nothing beyond what column-parallel
already does in compute-bound regimes (libquadmath ops dwarf any
cache effect). Useful primarily for nrhs ≥ nthreads where the
sub-solve overhead is amortized.

xtrsv_blocked is *not* race-free under any partition: the diagonal
walk has a tight cross-block dependency (block k's sub-solve reads
block k-1's trailing-update output). The two barriers per step are
structurally required.

### Rules

39. **Don't cache `omp_get_max_threads()`.** The ICV read is cheap
    relative to any BLAS call. A cache pins the value at first
    read — fine in production where OMP_NUM_THREADS is static, fatal
    in tests that alternate `omp_set_num_threads` and a silent
    perf bug if anything else also changes ICVs mid-run.

40. **One parallel region per call, not per inner loop.** When a
    serial outer drives a parallel inner, replace N/nb separate
    `#pragma omp parallel` regions with one region wrapping the
    whole walk + manual partition + barriers. Same sync cost order
    but threads stay pinned (cache locality) and there's no nested
    OMP. Requires `*_serial_` entries on the inner callees.

41. **No OMP-using function called from inside an OMP region.**
    Either expose a `_serial_` variant of the callee, or guard the
    callee with `omp_in_parallel()` and ensure its single-thread
    fallback is correct. Nested parallelism is fragile across
    libgomp / iomp / runtime configs; "outer parallel × inner
    parallel" is not a portable design.

42. **Column-partition is naturally race-free for SIDE='L'
    blocked-TRSM.** Each thread owns disjoint columns of B; sub-solve
    and trailing both touch only its slice. No inner barriers needed.
    SIDE='R' (or any algorithm with a cross-column dependency) does
    *not* have this property.



## Addendum 30: full-overlay re-sweep — systematic catalog (2026-05-20)

### Motivation

After the kind16 q-kernel refactors (qgemm 2D, qgemv slice/serial,
qtrsv blocked, qtrsm fast-path/blocked; Addenda 27-29 ported to the
q-prefix siblings), re-ran the full perf sweep at OMP=1 across all
three precision targets (kind10 e\*/y\*, kind16 q\*/x\*, multifloats
m\*/w\*) to confirm no regressions and to systematically map every
remaining sub-parity cell to a category.

### Methodology

- `scripts/run_perf_sweep.sh` at OMP=1, pinned to P-core 0, TIMEOUT=120s
  per perf binary, BLAS_PERF_ITERS=200 default.
- Routines that hit the 120s timeout (large libquadmath kernels: qgemm,
  qgemmtr, xtrsm, xtrmm, etc.) re-run on core 1 with reduced
  size×iters, captured into `/tmp/probe-timeouts*.tsv`.
- Filter: `ratio < 0.95` with `iters ≥ 20` (drops short-bench noise).

### Coverage

Final TSV: ~4,500 cells across 170 routines.
- kind10: 65 routines, 2,548 cells.
- kind16: 65 routines, ~1,560 cells (sweep + probe).
- multifloats: 41 routines, ~620 cells (sweep + probe; w\* still in flight).

### Result: zero new fixable cases

Every sub-parity cell maps to a previously-documented category.
Bucketed worst-per-routine view:

```
Cluster                                Cells      Cause / Addendum
-------------------------------------- ---------- ---------------------------
kind10 esbmv strided                   64         Addendum 22 (gcc outer-loop)
kind10 esymv strided                   62         Addendum 22
kind10 espmv strided                   57         Addendum 22
kind10 ygbmv T strided                 16         Addendum 24 (short banded)
kind10 etrmv L/strided borderline      9          Addendum 24 / Rule 22
kind10 etbmv banded                    6          Addendum 24
kind10 ytrsv L*N N=64 outliers         5          single-cell noise
kind10 ytpsv L/U borderline            5          Addendum 22 family (banded)
kind10 ytbsv strided                   4          Addendum 24
kind10 easum N≥1024                    4          memory bandwidth limit
kind10 ydotu/ygemv/eger small-N        2-3 each   single-cell noise
kind16 qtbmv/qtbsv T-branch banded     2-3 each   Addendum 24
kind16 qsymv/qspmv strided             2-3 each   Addendum 22 family
kind16 qtpmv/qtpsv N=64 outliers       2-3 each   single-cell noise
kind16 xqscal erratic ratios           3          short-workload variance
kind16 xgerc/xtrmv N=64                1-2 each   single-cell noise
multifloats mswap N=512/4096           2          memory bandwidth (L1 swap)
```

The big numerical buckets (s\*mv strided in kind10, banded-MV in kind10
and kind16) are accounted for by the two known intractability addenda
(22 and 24). The rest are single-cell N=64 variance — at iters≤20 a
single bad-luck timing dominates the ratio, and neighboring sizes for
the same routine sit at parity or above.

### What this confirms

- **The May 18 (Addendum 18) and May 19 (Addendum 19) trsv fixes are
  still in place.** kind10 etrsv L\* (LNN/LNU/LTN/LTU full set) and
  ytrsv U-T/L-T all show 0.93-1.16× on the re-probe.
- **My recent q-kernel refactors maintain or beat parity at OMP=1.**
  qgemm 0.97-1.04×, qgemv 0.91-1.03×, qtrsv 1.03-1.12× (UNN/UNU N=256
  the blocked path's single-region cache locality beats migrated
  unblocked), qtrsm 0.89-1.01× (one iters=3 outlier).
- **Multifloats overlay dominates migrated** — mgemm 20-25×, mgemmtr
  3-10×, msymv 3.6× across all stride combos. The SIMD double-double
  kernels (MGEMM_SIMD_MR=4, MBLAS_SIMD_DD) deliver the headline win.

### What remains (and why it stays)

The intractability stories from Addenda 22 and 24 still hold for the
two big buckets:

- **s\*mv strided** (~180 cells across kind10/kind16): combined
  AXPY+DOT inner with strided y. gcc's outer-loop setup costs ~5-10%
  more cycles than gfortran's. `fld %st(0)` is already present, asm
  inner instruction count matches. Closing the gap requires
  vector-intrinsic / asm-level work outside the C source.

- **Banded triangular T/C variants** (~30 cells): banded inner is
  bounded by KL+KU+1, typically ~33 elements at the perf bench's
  defaults. K-unroll-by-2 bookkeeping eats the latency saving on
  inners that short. Same constraint applies to qtbmv, qtbsv, etbmv,
  ytbmv, ytbsv.

### Rule

43. **Before opening a new investigation on a sub-parity cell:**
    (a) Check the worst-per-routine bucket count. If the routine has
    1-3 sub-parity cells in a sweep of ~20-100 cells per routine, it
    is almost certainly small-N variance — re-bench at iters≥100 first.
    (b) Match the cluster shape against Addenda 22 (combined inner +
    strided fallback) and 24 (banded inner length ≤ K+1). If it fits,
    the structural ceiling is real and the documented "tried and
    reverted" experiments apply — don't redo them.
    (c) Only after both filters miss is the cell worth a fresh
    asm-level investigation.

## Addendum 31: applying Rule 43 — etrmv LNN J-unroll-by-2 (2026-05-20)

### The cluster Rule 43 surfaced

After committing Addendum 30, re-read the sub-parity catalog with the
new "filter-before-investigate" lens. `kind10 etrmv` had 9 sub-parity
cells across 100+ cells, with this concentration:

```
etrmv LNN N=128:   0.905x
etrmv LNN N=1024:  0.935x
etrmv LNN/x2 N=1024: 0.911x
etrmv LNU N=128:   0.949x       (3-cell unit-stride cluster)
```

- **Rule 43(a)**: 3-4 cells per UPLO×TRANS combo is not single-cell
  noise — multiple sizes and stride variants all show the same gap.
- **Rule 43(b)**: doesn't match Addendum 22 (not a combined AXPY+DOT
  strided fallback — this is a plain AXPY inner) or 24 (full-row inner
  bounded by M, not by K+1).

So per the rule itself: investigate.

### What was missing

`etrmv` had a J-unroll-by-2 on UNN that the in-source comment described:
"Without this, UNN at N=1024 sat at 0.58x of migrated." The sibling LNN
path — same algorithmic structure, just j walking backward instead of
forward — had been left as the simple single-column reference loop:

```c
for (int j = N - 1; j >= 0; --j) {
    const T temp = x[j];
    if (temp != zero) {
        const T *aj = &A_(0, j);
        for (int i = N - 1; i > j; --i) x[i] += temp * aj[i];
    }
    if (nounit) x[j] *= A_(j, j);
}
```

The pairing logic was less obvious because j descends, but iteratively
the same property holds: at any pair (j, j-1), both x[j] and x[j-1]
are pristine on entry to iter j (iter j's inner only touches i>j).
So saved into t0/t1, the trailing-rows inner can service both column
contributions in one pass, with boundary handling at i=j and i=j-1.

### Result

```
etrmv LNN N=128:  0.905 → 1.813x  (+100% over migrated)
etrmv LNN N=256:  0.992 → 1.606x
etrmv LNN N=512:  ~1.0  → 1.605x
etrmv LNN N=1024: 0.935 → 1.481x
etrmv LNU N=128:  ~1.0  → 1.625x
etrmv LNU N=1024: ~1.0  → 1.552x
```

Fuzz 80/80 pass (max err 3.8e-19, within tolerance).

### Rule 43 worked

The category that survived Addendum 30's bucketing — multi-cell cluster
that didn't match either intractability addendum — was genuinely a
missing-sibling-fix case, fixed with the same recipe already applied
to the upper-triangular path. The rule is doing its job: it flagged
exactly the one cluster that hadn't been investigated and skipped the
~250 cells already accounted for.

### Follow-up

Surveyed siblings: `ytrmv` (kind10 complex) LNN/UNN run at parity
without J-unroll — gfortran handles `_Complex long double` AXPY
inners differently and the C reference already matches. `qtrmv`/
`xtrmv` (kind16) are libquadmath-bound — per-op cost dominates and
J-unroll doesn't help at that rate. So the etrmv fix is kind10-real-
specific and complete.

## Addendum 32: TRSM SIDE='R' — row-partition OMP closes the only L3 OMP gap (2026-05-21)

### What the OMP scaling sweep surfaced

After the perf re-sweep landed (Addendum 30), I ran a separate OMP
scaling sweep (`overlay@OMP4 / overlay@OMP1`) across the L3 routines
that have OMP code. The L3 column-partition pattern landed in
Addenda 27-29 covered SIDE='L' uniformly, but the sweep showed a
flat band of "doesn't parallelize" cells concentrated in a single
shape:

```
etrsm  RLN/RUN/RLT/RUT  M=256-512  scaling ≈ 0.99-1.04x
ytrsm  RLN/RUN/RLTC/RUTC + conj  scaling ≈ 0.99-1.05x
mtrsm  RLN/RUN/RLT/RUT  scaling ≈ 1.00x  (the diag SIMD core is fast
                                          but the outer was serial)
wtrsm  RLN/RUN/RLTC/RUTC          scaling ≈ 1.00x
```

kind16's `qtrsm`/`xtrsm` already had the wrapper — the macro was
added during the Addendum 28-29 work but only kind16 entry points
got it. SIDE='L' for all eight routines had been column-parallel
since Addendum 27; SIDE='R' had been left serial because the
algorithm's natural axis (j over columns of B) walks the diagonal
serially and doesn't admit a column-partition.

### Why SIDE='R' partitions cleanly on rows

For X · op(A) = α·B the j loop is the diagonal walk and must run
sequentially — each column of B depends on the previously-solved
columns. **But every row of B is processed identically and
independently.** Splitting M across threads gives disjoint row slices
of B; the shared `a` is read-only; no barriers are needed inside the
parallel region.

In column-major storage a row slice `[i_lo, i_hi)` is simply the
pointer-shifted submatrix starting at `b + i_lo` with leading dim
`ldb` and `Mslice = i_hi - i_lo` rows. The cores' inner loops over
M just operate on the shifted view.

### The kind10 wrapper

To re-use the existing cores I changed their leading signature from
`(M, ...)` to `(i_start, i_end, ...)` (a row range) and added a
single macro that does the partition once:

```c
#define ETRSM_OMP_WRAP_R(name, core)                                     \
    static void name(int M, int N, T alpha,                              \
                     const T *a, int lda, T *b, int ldb, int nounit) {   \
        if (M >= ETRSM_OMP_N_MIN && blas_omp_max_threads() > 1           \
                                && !omp_in_parallel()) {                 \
            _Pragma("omp parallel") {                                    \
                int tid = omp_get_thread_num();                          \
                int nt  = omp_get_num_threads();                         \
                int is  = (int)((long long)M * tid / nt);                \
                int ie  = (int)((long long)M * (tid + 1) / nt);          \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);          \
            }                                                            \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }         \
    }
```

`ytrsm`'s TC variants carry an extra `conj_flag` arg, so a sister
macro `YTRSM_OMP_WRAP_R_TC(name, core, cflag)` invokes the core
with that constant baked in.

### The multifloats wrapper

mtrsm/wtrsm route SIDE='R' through `mtrsm_simd_diag_R` /
`wtrsm_simd_diag_R`, which process 4 rows of B per SIMD-DD vector
step. Naïve row-partitioning would break alignment on interior
threads, so the wrapper rounds slice boundaries down to multiples
of 4 for all but the last thread:

```c
#pragma omp parallel if(use_omp)
{
    int tid = ..., nt = ...;
    int i_lo = (int)((long long)M * tid / nt);
    int i_hi = (int)((long long)M * (tid + 1) / nt);
    if (tid > 0)      i_lo &= ~3;
    if (tid < nt - 1) i_hi &= ~3;
    const int Mslice = i_hi - i_lo;
    if (Mslice > 0) {
        T *b_slice = b + i_lo;
        mtrsm_simd_diag_R(op, Mslice, N, alpha, a, lda, b_slice, ldb, nounit);
    }
}
```

The last thread absorbs the `M & 3` tail (its `i_hi` is not rounded),
so the SIMD path stays whole on interior threads and only the trailing
thread takes a scalar-cleanup hit — which the existing `*_simd_diag_R`
already handled internally.

### Result (overlay vs migrated, OMP=4)

| Routine       | Before    | After                  |
|---------------|-----------|------------------------|
| etrsm R\*     | ~0.99x    | 3.56-3.87x             |
| ytrsm R\* (12 keys)   | ~0.99x | 3.58-3.97x          |
| mtrsm R\*     | ~1.02x    | 17-21x (overlay 3.85x internal scaling, on top of SIMD-DD's single-thread advantage) |
| wtrsm R\* (12 keys)   | ~1.04x | 15-22x (3.62-3.95x internal scaling) |

Fuzz 80/80 pass on all four routines.

### `!omp_in_parallel()` is load-bearing

Each wrapper gates on `M >= *_OMP_N_MIN && blas_omp_max_threads() > 1
&& !omp_in_parallel()`. The third guard is what makes this safe to
expose. If a caller is already inside an OMP region (rare but
possible via `*symm`/`*syr2k` paths that fan out to a trsm), the
wrapper falls through to the single-threaded core — no nested
parallelism, per the project-wide rule.

### Coverage now uniform

After this fix the L3 OMP map is uniform: SIDE='L' uses column-
partition, SIDE='R' uses row-partition, and every routine in
{e,y,q,x,m,w}trsm has both. The OMP scaling sweep is now flat at
~3.5-4.0x across L3 (Amdahl-capped at 6 P-cores).

## Addendum 33: Add-30 catalog re-verification — Rule 30 catches the rest (2026-05-21)

### Motivation

Stop-hook flagged that Addendum 30's "filter-before-investigate"
Rule 43 had let ~250 cells be dismissed by category match instead of
re-verified per-routine. Walked the catalog again, this time with
Rule 30 discipline (≥5 trials, iters≥20000 at small N), to confirm
each was actually structural and not just bench noise.

### What changed when re-bench discipline was applied

Bench iters mattered more than expected:

```
Routine cluster                  iters=200    iters=1000    iters≥20000 (5+ trial)
                                 (Add-30)     (1-trial)     (median)
─────────────────────────────────────────────────────────────────────────────────
esbmv unit-stride L N=128        0.775        0.995         1.04
esbmv unit-stride U N=128        0.878        1.035         1.02
esbmv strided (all sizes)        0.91         0.86-0.95     0.89-0.93  ← structural
espmv unit-stride L N=128        0.768        1.05          1.04
espmv strided (all sizes)        ~0.90        0.86-0.95     0.89-0.92  ← structural
esymv strided                    0.93         0.84-0.96     0.89-0.93  ← structural
etbmv UNN N=128                  0.92         0.92          0.97
etbmv LTN N=128                  0.95         0.37          0.99       ← was pure noise
etbmv UTU N=512                  0.95         0.93          1.10
ygbmv N N=128                    0.95         0.83-0.89     0.99       ← was noise
ygbmv T/x-1/y-1 N=512            0.95         0.91          0.94       ← structural
ytbsv all                        0.95         0.84-0.99     0.94-1.0
ytpsv L/U                        0.93         0.95-1.0      1.0
easum                            0.95         (ratio inverted; see below)
```

The pattern: **iters=200 catches one class of noise, iters=1000 catches
a different class, only iters≥20000 with 5+ trials reveals the true
distribution**. Cells that look structural at iters=1000 can be at
parity at iters=20000 (etbmv LTN N=128 went from 0.37 to 0.99). Cells
at parity at iters=1000 can be sub-parity at iters=20000 (the s\*mv
strided ~10% gap is more visible at high iters where the inner-loop
overhead amortizes).

### A genuine fix surfaced: esymv/qsymv strided index-recompute

The re-walk caught an anti-pattern in `esymv` and `qsymv` strided
fallbacks that the iters=200 noise had been masking. Both recomputed
`kx + k*incx` and `ky + k*incy` inside the inner loop:

```c
for (int k = i + 1; k < N; ++k) {
    y[ky + k * incy] += temp1 * A_(k, i);   ← multiply per iter
    temp2 += A_(k, i) * x[kx + k * incx];
}
```

The Netlib Fortran reference (`DSYMV`) uses incrementing IX/IY:

```fortran
DO 90 I = 1,J - 1
    Y(IY) = Y(IY) + TEMP1*A(I,J)
    TEMP2 = TEMP2 + A(I,J)*X(IX)
    IX = IX + INCX
    IY = IY + INCY
   90 CONTINUE
```

Sister routines `esbmv` and `espmv` already use the increment-based
form. Converting esymv/qsymv to match gave a modest but consistent
win on L-branch unit-stride (esymv L N=256 went from 0.949 → 1.00+).
Strided cases stayed at the Addendum 22 ceiling — that gap is in the
outer-loop setup, not in the inner.

Committed as `parallel-blas: esymv/qsymv strided — increment-based ix/iy`.

### easum was mis-classified as bandwidth-bound

Addendum 30's table listed "kind10 easum N≥1024 — memory bandwidth
limit (4 cells)". Re-bench at iters=2000+ shows overlay is **4-9×
faster** than migrated (ratio = `t_mg/t_ov` so values >1 mean overlay
wins):

```
easum N=64    ratio = 4.27
easum N=512   ratio = 5.59
easum N=1024  ratio = 9.88
easum N=4096  ratio = 8.89
```

The original sweep's iters=200 caught variance-dominated cells where
absolute timings were too short to discriminate the two
implementations. With proper iters the overlay's `restrict`-annotated
loop crushes the migrated build's aliased reads. **Not sub-parity;
remove from the catalog.**

### What remains genuinely sub-parity (and why it stays)

After Rule 30 discipline applied per-cell, the surviving sub-parity
cells fall into exactly two documented categories:

1. **s\*mv strided fallback (Add-22 family, ~180 cells across kind10
   esbmv/espmv/esymv and kind16 qsymv/qspmv).** ~10% gap, 5-trial
   median 0.89-0.93. The outer-loop setup overhead (one extra mul per
   index per outer iter) is gcc-vs-gfortran codegen ceiling for
   strided long-double, not addressable from C. Documented in
   Addendum 22 with asm proof.

2. **ygbmv T/x-strided (Add-24 follow-up, ~14 cells).** 10-trial
   median 0.94. The outer-loop dep chain (alpha*s reduction at the
   end of each outer iter) bounds the throughput; the inner-loop asm
   already wins. Tag-and-stop per the Add-24 follow-up disposition.

Every other cell from the Addendum 30 catalog — 6 of 8 listed clusters,
~70+ cells — was bench noise. After Addendum 25 / Rule 30 (≥20000
iters at small N, ≥5 trials), they are at parity or better.

### Rule

44. **Rule 30 sweeps still need 5+ trials at iters≥20000 to declare
    structural sub-parity.** The original Add-30 sweep used iters=200
    and read 1-3 trial values, mis-classifying ~70 cells as
    structurally sub-parity when they were variance. **Filter-by-
    category (Rule 43) is not a substitute for re-bench-by-cell.**
    Cluster-shape matching tells you what tradition of failure a
    cell *might* fit; only a 5-trial high-iter rerun tells you which
    cells are *actually* sub-parity.

## Addendum 34: egemm — single-region + omp single Bp + omp for ic restores scaling (2026-05-21)

### Context

egemm's blocked path (NN, NT, TT — i.e. anything that isn't the
TA='T' TB='N' fast path) was structurally serial whenever N ≤ NC.
The pre-fix decomposition wrapped the whole call in `#pragma omp
parallel` and then `#pragma omp for schedule(static)` over `jc`
(the N-band). With default NC=512, any N ≤ 512 produces a single
jc iteration, which `omp for` hands to thread 0 — every other
thread idles. Measured OMP=4/OMP=1 scaling at N=512 was 0.98×.
N=1024 (2 jc bands) saw 1.90×. Only the column-parallel TN fast
path (which loops over j2 ∈ [0, N), not NC-bands) scaled cleanly.

Findings note already flagged this as untested ground (line 205,
"Egemm + ygemm at OMP>1 — current numbers are OMP=1. Threading is
the actual point of the overlay"). xgemm hit the identical pattern
and fixed it via `collapse(2)` in Add-27 (Rule 35).

### Experiment

Two refactors in sequence:

1. **First pass: collapse(2) over (jj, ii) tile coordinates.**
   Same shape as Add-27. Each thread owns Ap+Bp scratch, packs
   both per (own tile, pc). Big OMP=4 win at large N (3.6×
   scaling at N≥512), but OMP=1 regressed 3-6% because each
   tile re-packs Bp instead of once per (jc, pc), and small-N
   parallelism was still limited by tiles-per-call.

2. **Second pass: single outer `omp parallel`, shared Bp, `omp
   single` packs it once per (jc, pc), `omp for` over ic.**
   Each thread keeps a private Ap. The two implicit barriers
   (after `single` and after `for`) are the cheap synchronisation;
   Bp is read-only during the `for`. Removes the redundant
   packing and the OMP=1 regression with it.

### Result

Final OMP=4/OMP=1 scaling (5.5 GFs to 11 GFs absolute throughput
at N=1024):

| key | N=128 | N=256 | N=512 | N=1024 |
|-----|-------|-------|-------|--------|
| NN  | 1.02× | 1.30× | 3.44× | 3.60× |
| NT  | 0.97× | 1.17× | 3.62× | 3.66× |
| TT  | 0.93× | 1.33× | 3.31× | 3.64× |

OMP=2/4/8 at N=512 NN: 1.82× → 3.44× → 3.77× (i7-8700 has 6
P-cores, so 4-8 thread asymptote is expected). OMP=1 numbers
within 1-2% of pre-fix — no real regression.

Small-N parallelism is bounded by `M / MC` tiles. At N=128 M=128
MC=64 → 2 ic tiles → 2-way maximum. At N=256 M=256 → 4 ic tiles →
4-way maximum. The 1.3× actually measured at N=256 OMP=4 leaves
~3× on the floor — likely heap-lock contention on the per-thread
Ap alloc + Bp shared-cache contention; deferred since it's a
small-problem regime and the large-N wins are the primary target.

Versus migrated egemm (perf_egemm output, OMP=4):

| key | N=512 | N=1024 |
|-----|-------|--------|
| NN  | 5.46× | 6.52× |
| NT  | 5.76× | 6.82× |
| TT  | 8.74× | 13.19× |

### Rules

45. **For GotoBLAS-style packed kernels with shared B-packing per
    (jc, pc): single outer `omp parallel`, `omp single` to pack
    Bp once, `omp for` over ic.** The implicit barriers are
    cheaper than per-(jc, pc) fork-join, and Bp shared across
    threads avoids the redundant per-tile packing that a naive
    collapse(2) forces. Apply when Bp dominates per-tile setup
    cost (true at any KC × NC large enough to matter).

46. **`#pragma omp parallel` + `#pragma omp for` over the *first*
    loop is a trap when later loops are larger and the first
    loop is short.** The original egemm parallelised jc (1
    iteration at N ≤ NC). Same anti-pattern as the pre-Add-27
    xgemm. Whenever `for_axis_iters < nthreads`, the `omp for`
    *cannot* parallelise — pick a different axis or use
    `collapse`.

## Addendum 35: ygemv N-branch — J-unroll the OMP path (2026-05-21)

### Context

ygemv (kind10 complex GEMV) N-branch scaled to only 2.7-2.8× on
OMP=4. T and C branches scaled to 3.5-3.9×. The structural
difference: N-branch is a rank-1 update, so each y[i] is RMW'd
once per column j. With M*N writes to y across iters, the parallel
version saturates L3/y-traffic before reaching peak compute.

egemv (real GEMV) has J-unroll-by-2 on the N-branch's OMP path
(pre-existing — see code), and scales to 3.0-3.5×. ygemv didn't
adopt the same trick because a previous comment claimed:
"J-unroll-by-2 (2 cmuls per iter = 8 fmuls + 4 fadds) hits stack
pressure and gcc spills a1 to memory, *re-loading* each fmul
operand". That claim was measured at OMP=1, where the spill cost
dominates the (already-cheap) y RMW.

Under OMP=4 the calculus changes: y traffic is the bottleneck
(memory-bandwidth-limited), so halving y reads/writes by reusing
each y[i] load for two consecutive columns outweighs the spill
overhead.

### Experiment

Branch on use_omp in C source:
- OMP=1: keep the single-column inner (gfortran-style, no spill).
- OMP>1: J-unroll-by-2 — `y[i] = (y[i] + t0*a0[i]) + t1*a1[i]`,
  one y RMW per pair of columns.

### Result

ygemv N-branch OMP=4 scaling (overlay GFs, scaling vs OMP=1):

| N    | before | after | abs GFs (before → after) |
|------|--------|-------|--------------------------|
| 128  | 2.48×  | 2.65× | 6.24 → 7.55 |
| 256  | 2.75×  | 3.29× | 7.51 → 9.54 |
| 512  | 2.83×  | 3.46× | 7.51 → 9.80 |
| 1024 | 2.70×  | 3.14× | 6.73 → 8.09 |
| 2048 | 2.66×  | 2.77× | 6.53 → 7.19 (DRAM bw) |

30% absolute throughput improvement at N=512. OMP=1 path unchanged
(within bench noise). Brings N-branch parity with the T/C-branch
scaling profile.

N=2048 stays at ~2.77× because A (128 MB) exceeds L3 (12 MB) —
DRAM-bandwidth bound regardless of compute pattern.

### Rules

47. **Apply J-unroll-by-2 (or equivalent y-reuse trick) to GEMV-N
    OMP paths even when it regresses serial — the parallel
    bottleneck is y-traffic, not register pressure.** Standalone
    serial cost from x87 spill is small (a few percent); halving
    y-RMW traffic at OMP=4 is a 20-30% win. Use `if (use_omp)`
    in C source to split serial-optimal from parallel-optimal
    inner-loop forms.

## Addendum 36: esymv / yhemv — per-thread y_priv + reduction unlocks SYMV/HEMV scaling (2026-05-21)

### Context

esymv and yhemv had `ESYMV_OMP_MIN` / `YHEMV_OMP_MIN` macros defined
but *zero* `#pragma omp` directives — completely serial. The
broad OMP=1 vs OMP=4 sweep flagged both at ~1.0× scaling.

Why no OMP existed: the Netlib two-pass form walks A column-by-
column (stride-1) and writes y[k] for k > j (L) or k < j (U) on
each j. Trying to parallelise the outer j loop races on y[k]
writes since multiple threads' j-ranges produce overlapping k
output ranges.

### Experiment

Per-thread private y buffer + reduction:
1. Allocate `y_priv[nt][N]`.
2. Each thread zeros its own y_priv slice.
3. `#pragma omp for schedule(static, 1)` over j — interleaves
   columns across threads to balance the triangular work (per-
   column work is `(N - j)` for L, `j` for U — linear gradient).
4. Implicit barrier at end of `omp for`.
5. `#pragma omp for schedule(static)` reduces y_priv across
   threads into y.

Same `omp parallel` region wraps both phases; second barrier
gates the reduction.

`ESYMV_OMP_MIN` / `YHEMV_OMP_MIN` bumped to 128 (work too small
below that to amortise the alloc + 2 barriers).

### Result

esymv (overlay GFs at unit stride, OMP=4/OMP=1 scaling):

| key | N=128 | N=256 | N=512 | N=1024 |
|-----|-------|-------|-------|--------|
| U   | 2.78× | 3.57× | 3.76× | 3.71× |
| L   | 2.46× | 3.42× | 3.67× | 3.85× |

yhemv:

| key | N=128 | N=256 | N=512 | N=1024 |
|-----|-------|-------|-------|--------|
| U   | 4.42× | 5.00× | 5.29× | 5.00× |
| L   | 3.29× | 4.14× | 4.39× | 4.27× |

yhemv U exceeds 4× on a 4-thread test — likely because the
serial form was already mildly sub-parity and the parallel form
also gained per-thread compute density (per-column work fits in
L1 with no thread contention).

Both fuzz tests clean (80/80, max ulp errors at machine
precision).

### Debugging note

First version used `nowait` on the work-distribution `omp for`,
intending only the parallel-region close to barrier. fuzz
diagnosed real numerical drift (4 of 80 cases at random N
between 158-236). The reduction `omp for` started before all
threads had finished writing y_priv. Removed `nowait` — `omp
for`'s implicit end-of-region barrier is exactly what gates
the reduction.

### Rules

48. **SYMV / HEMV: per-thread y_priv + reduction is the
    standard parallelisation, even though it costs `nt*N` extra
    memory + a reduction pass.** Column-walk (stride-1) is
    structurally required for memory access; trying to
    parallelise the outer loop without privatisation races on
    the cross-diagonal y writes. The reduction overhead is `N
    * nt` adds — trivial vs `N²/2` cmadds.

49. **`schedule(static, 1)` is the right load balance for
    triangular column work.** Per-column work is linear in
    `(N - j)` (L) or `j` (U). Round-robin chunk size 1 gives
    each thread an interleaved set of columns covering the
    full range, so heavy and light columns are evenly mixed.
    `schedule(static)` (no chunk) gives thread 0 the heaviest
    block, last thread the lightest — 2-3× imbalance.

50. **Don't use `nowait` on the work-distribution `omp for`
    when the next pragma reads cross-thread output.** The
    implicit barrier at end of `omp for` is what protects the
    next pragma from racing on partially-written data. `nowait`
    only after the final consumer.

## Addendum 37: etrmv / ytrmv — out-of-place buffer dissolves the in-place dependency (2026-05-21)

### Context

etrmv (real triangular MV) and ytrmv (complex triangular MV)
had no OMP. Headers commented "inherently serial over j (each j
writes to x[j] or reads x[i] from a region that earlier j's
modified)". True only for the *in-place* algorithm: x := A·x
updates x while iterating, so the loop order matters.

Using a temporary output buffer dissolves the dependency, after
which both transposes parallelize cleanly.

### Experiment

Two distinct parallel patterns depending on TRANS:

- **TR='T' or 'C'**: each j produces a single output element
  x[j] (dot product of column j with the trailing x). Different
  j's write to different x[j] — no overlap. Single shared
  `y_buf[N]`; each thread writes its assigned j-entries; final
  `omp for` copies `y_buf` back to x.

- **TR='N'**: rank-1 column update — each column j contributes
  to x[i] for i > j (L) or i < j (U). Cross-thread j ranges
  produce overlapping i write ranges. Per-thread `y_priv` +
  reduction (esymv pattern, Add-36).

`schedule(static, 1)` for triangular load balance per Rule 49.

### Result

etrmv OMP=4 vs OMP=1 scaling (was ~1.0× everywhere):

| key  | N=128 | N=256 | N=512 | N=1024 |
|------|-------|-------|-------|--------|
| UNN  | 2.60× | 3.50× | 4.31× | 4.47× |
| UTN  | 1.44× | 3.36× | 4.07× | 3.65× |
| LNN  | 2.63× | 3.72× | 4.09× | 4.53× |
| LTN  | 1.21× | 2.92× | 3.91× | 4.02× |

ytrmv (complex):

| key  | N=128 | N=256 | N=512 | N=1024 |
|------|-------|-------|-------|--------|
| UNN  | 3.15× | 3.46× | 3.95× | 3.86× |
| UTN  | 2.21× | 3.74× | 3.67× | 3.44× |
| UCN  | 2.52× | 3.62× | 3.90× | 3.56× |
| LCU  | 2.46× | 3.16× | 3.70× | 3.61× |

OMP=1 path unchanged (use_omp short-circuit).
Fuzz clean for both (80/80 at machine precision).

### Rules

51. **"Inherently serial" comments deserve a fact-check.** The
    in-place dependency on x is dissolved by allocating a
    separate output buffer; the cost is a buffer alloc + copy
    pass, often <5% of the parallel win. Read the comment
    carefully — "inherently serial in-place" is a different
    claim from "inherently serial".

## Addendum 38: comprehensive OMP=1 vs OMP=4 sweep — May 2026 status (2026-05-21)

### Method

Bench every kind10 perf binary at OMP=1 and OMP=4 with default
size set, unit stride, iters=50 (skip strided cells). Scaling =
OMP4_GFs / OMP1_GFs per (routine, key, N). Per-routine median
across all cells.

### Result after Add-34/35/36/37 + L2 rank-1 hygiene

Sorted by median scaling, ascending:

| routine  | n  | median | min  | max  | category               |
|----------|----|--------|------|------|------------------------|
| ecopy    | 4  | 0.99×  | 0.98 | 1.00 | L1 memcpy, mem-bw      |
| easum    | 4  | 1.00×  | 1.00 | 1.00 | L1                     |
| eaxpy    | 4  | 1.00×  | 1.00 | 1.01 | L1                     |
| edot     | 4  | 1.00×  | 1.00 | 1.00 | L1                     |
| eswap    | 4  | 0.99×  | 0.99 | 1.06 | L1                     |
| escal    | 4  | 1.00×  | 0.88 | 1.05 | L1                     |
| erot     | 4  | 1.00×  | 0.96 | 1.03 | L1                     |
| esyr     | 8  | 1.60×  | 1.22 | 1.88 | L2 sym rank-1, mem-bw  |
| yher     | 8  | 1.64×  | 1.47 | 1.89 | L2 Herm rank-1         |
| etrsv    | 32 | 1.68×  | 0.93 | 3.31 | L2 — Amdahl, Add-28    |
| ygerc    | 4  | 1.96×  | 1.67 | 2.08 | L2 rank-1, mem-bw      |
| ygeru    | 4  | 2.01×  | 1.66 | 2.38 | L2 rank-1, mem-bw      |
| egemm    | 12 | 2.41×  | 0.97 | 3.81 | L3 — Add-34            |
| esyrk    | 12 | 2.38×  | 1.20 | 3.91 | L3                     |
| egemmtr  | 24 | 2.45×  | 1.15 | 3.87 | L3                     |
| etrmv    | 32 | 2.66×  | 0.99 | 4.18 | **L2 — Add-37**        |
| egemv    | 8  | 3.22×  | 2.16 | 3.85 | L2 — Add-34            |
| ygemv    | 12 | 3.56×  | 3.06 | 4.03 | L2 — Add-35            |
| esymv    | 8  | 3.64×  | 2.47 | 4.28 | **L2 — Add-36**        |
| eger     | 4  | 3.64×  | 1.54 | 5.93 | L2 rank-1              |
| ytrmv    | 48 | 3.59×  | 2.17 | 9.92 | **L2 — Add-37**        |
| esymm    | 12 | 3.71×  | 1.85 | 4.00 | L3 (no change needed)  |
| yhemv    | 8  | 3.70×  | 2.93 | 3.99 | **L2 — Add-36**        |
| yherk    | 12 | 3.09×  | 1.94 | 3.86 | L3                     |
| ysyrk    | 12 | 3.07×  | 2.21 | 3.94 | L3                     |

### Sub-2× routines remaining (and why)

- **L1 BLAS** (ecopy, eaxpy, easum, edot, escal, eswap, erot):
  memory-bandwidth bound. OMP can't help — DRAM doesn't get
  faster per thread. Status: accept.

- **esyr / yher** (sym/Hermitian rank-1, ~1.6× median):
  memory-bandwidth bound on the triangular A writes. Same
  memory ceiling as the L2 GER family. Status: accept.

- **etrsv** (median 1.68, max 3.31): structural Amdahl ceiling
  M/nb on the blocked path (Add-28). At small N the blocked
  threshold fails over to serial; at large N it scales to 2.7-
  3.3×. Status: known structural limit, doc'd.

- **ygerc / ygeru** (~2.0× median): same memory-bw story as
  GER; rectangular A writes dominate. Status: accept.

### Sub-3× routines with structural fixes possible

- **esyrk / egemmtr** (median 2.38 / 2.45): outer jc-parallel
  with serial inner egemm. At small N (jc loop count = N/nb is
  low) parallelism is bounded. 2D `collapse(jc, gemm M-tile)`
  would help small-N. Deferred — small-N regime is bounded
  loss and complexity is high.

- **egemm** (median 2.41, max 3.81): Add-34's M/MC tile limit
  caps small-N. At N>=512 scales to 3.7×. Status: structural
  limit at small N.

### Big wins this session

Four routines went from completely unparallelised (1.0×) to
healthy scaling:

| routine | before | after | absolute @ N=1024 OMP=4 |
|---------|--------|-------|-------------------------|
| esymv   | 0.98×  | 3.64× | 6.8 GFs                 |
| yhemv   | 1.03×  | 3.70× | 9.4 GFs                 |
| etrmv   | 1.01×  | 2.66× | 4.5 GFs                 |
| ytrmv   | 0.99×  | 3.59× | 6.7 GFs                 |

Plus egemm/ygemv N-branch improvements from Add-34/35.

## Addendum 39: esyrk — full OpenBLAS DSYRK cooperative-kernel port (2026-05-21)

### Problem

esyrk's median OMP=1→OMP=4 scaling sat at 2.38× (Add-38 sweep).
Worst cells were at small N: UN/LN @ N=128 cliffed at ~1.2× because
the outer `#pragma omp parallel for schedule(dynamic, 1)` over fixed
`nb=64` jc-blocks gave 2 jc iterations at N=128 — only 2 of 4 (or
12) threads got work. Even at large N, the inner `egemm` call
allocated ~2 MB packed-B and ~256 KB packed-A on every block,
mmap-heavy.

### What OpenBLAS does (driver/level3/level3_syrk_threaded.c)

1. **Below threshold → serial.** `if (n < nthreads * SWITCH_RATIO)
   SYRK_LOCAL(...); return;`. SWITCH_RATIO is 8–32 depending on
   build. Below it, parallelism overhead dominates; just run
   serial.

2. **Quadratic N partition for cooperative work balance.** Width
   sequence `w_t = sqrt(i_t² + N²/nthreads) - i_t` is the *same*
   for LOWER and UPPER; the difference is purely how the array is
   filled. LOWER fills forward (thread 0 gets the widest band at
   the low end and dominates the diagonal triangle, no off-diag
   contribution); UPPER fills backward (thread 0 gets the
   narrowest band at the low end and contributes off-diag slabs
   to every higher-index thread). Either way each thread's total
   work — own diag triangle plus rectangles it produces using
   OWN-row-panel × OTHER-col-panel — equals N²/(2·nthreads).

3. **Each thread runs the GotoBLAS loop inline.** No recursive
   gemm call. Per K-chunk: pack own A row panel (`sa`), pack own
   A column panel sub-pieces (`buffer[bs]`), compute diagonal
   block via triangle-aware kernel, then consume OTHER threads'
   buffer pointers for off-diag rectangles.

4. **Lock-free buffer-sharing via per-(producer, consumer, bs)
   flags.** Each flag occupies one cache line. Producer writes
   pointer = "buffer ready"; consumer clears = "buffer consumed".
   Producer waits at top of next K iter for consumers to clear
   before reusing the buffer slot. `DIVIDE_RATE=2` sub-buffers
   enable producer/consumer pipelining.

5. **Pre-allocated buffers per thread.** sa and buffer_pool
   allocated once outside the K loop. No per-block malloc tax.

### Port

`src/parallel_blas/kind10/esyrk.c` rewritten — ~700 LOC. New
pieces:

- `pack_A_panel` / `pack_B_panel`: identical-pattern packers
  reused from egemm but renamed; pack A^op (with optional
  transpose) for row and col panels.
- `kernel_2x2` / `kernel_edge` / `macro_kernel_rect`: same MR=NR=2
  x87 stack-resident kernels as egemm.
- `macro_kernel_tri`: NEW. Triangle-aware variant for the
  diagonal block. Sub-tiles entirely below/above the diagonal
  use `kernel_2x2` / `kernel_edge`; sub-tiles crossing the
  diagonal fall back to entry-by-entry with UPLO check.
- `syrk_quadratic_partition`: width sequence above; LOWER fills
  forward, UPPER backward.
- `inner_syrk`: the per-thread cooperative body. PHASE 1 packs
  + diagonal; PHASE 2 work-steal own sa × LOWER threads' buffers
  (LOWER) or HIGHER threads' (UPPER); PHASE 3 iterates remaining
  own row chunks. Cleanup drain at end of K loop.
- Flag plumbing: `flag_at` indexer + `cpu_relax`/`WMB`/`RMB`
  helpers.
- Serial fallback kept verbatim for small N (`N < nthreads ·
  ESYRK_SWITCH_RATIO`) and OMP=1.

`ESYRK_SWITCH_RATIO` default 16, `ESYRK_MC=64`, `ESYRK_KC=256`,
`DIVIDE_RATE=2`. All env-overridable.

### Result

Fuzz: 80/80 cases pass at OMP=1/2/4/8/12, across 4 seeds = 960
cases clean at machine precision (max abs err ~5e-19).

Perf @ N=512, kind10 i7-8700 (6 P-cores + SMT, 12 threads):

| key | OMP=1  | OMP=4   | OMP=8   | OMP=12  | OMP=1 mig |
|-----|--------|---------|---------|---------|-----------|
| UN  | 2.45   | 10.72   | 9.84    | 15.63   | 1.25      |
| UT  | 2.14   | 10.81   | 10.81   | 16.44   | 2.16      |
| LN  | 2.39   | 10.75   | 11.24   | 13.15   | 1.24      |
| LT  | 2.20   | 10.78   | 11.35   | 13.05   | 2.13      |

OMP=1→OMP=4 scaling = ~4.4× (near-linear on 4 P-cores).
OMP=1→OMP=12 scaling = ~6.5× (SMT contention but still climbing).

esyrk's median OMP=1→OMP=4 scaling went from **2.38× to 4.4×**.
Absolute throughput at N=512 OMP=4 went from ~5.7 GFs to 10.7 GFs
(1.9× over the previous structure). At N=512 OMP=12 we hit
~16 GFs — ≈8× the migrated Fortran reference even at OMP=1, ≈12×
at the cooperative paths.

OMP=1 path is the serial blocked fallback (same code as before,
within noise — UN/LN @ N=64–128 at 0.88×–1.04× of pre-Add-39).

### Rules

52. **Recursive gemm calls inside a parallel block create
    serialization at the gemm packing layer.** Each thread
    independently allocates and packs its own Bp/Ap inside each
    block. Splitting one big GotoBLAS pass into N small ones
    multiplies the packing cost by N AND prevents any
    inter-thread sharing of the packed panels. If the trailing
    update is GEMM-shaped, inline the GotoBLAS loop in each
    thread instead — pre-allocate buffers once, and arrange
    flag-based packed-buffer sharing so each row of A is packed
    by exactly one thread per K-chunk.

53. **For SYRK-shaped output (only one input matrix), the
    cooperative pattern is: each thread produces both a row
    panel and a col panel of A, then uses OWN row panel ×
    OTHER thread's col panel for off-diagonal output cells.**
    Direction follows UPLO: LOWER ⇒ consume LOWER-index
    threads' buffers (their cols are at indices lower than
    own rows); UPPER ⇒ HIGHER-index. The own-diagonal
    contribution stays triangle-aware via a per-sub-tile
    UPLO check.

54. **Quadratic N partition `w = sqrt(i² + N²/nt) - i` is the
    right balance for the cooperative kernel — but the
    direction of fill is UPLO-dependent.** LOWER fills forward
    (thread 0 = wide leading band, low-index cols, dominated
    by diagonal); UPPER fills backward (thread 0 = narrow
    leading band, low-index cols, dominated by off-diagonal
    rectangles to higher-index threads). NEITHER UPLO mirrors
    the naive per-col workload — total per-thread work always
    includes off-diag contributions and only this asymmetric
    fill makes them sum equally.

55. **Lock-free flag protocol needs producer-side wait at the
    START of next K iteration, not the end.** Otherwise a
    fast consumer that already cleared the flag races against
    the producer overwriting its buffer for the next K chunk.
    OpenBLAS's pattern: producer waits for `working[i][bs] ==
    0` before packing buffer[bs] anew; consumer always
    clears flag AFTER kernel returns (with WMB). On x86 only
    a compiler barrier is needed for the producer wait → buffer
    write ordering.

### Addendum 39a: same packed kernel for the OMP=1 serial path

The cooperative kernel sits on a clean packed-MR×NR core. Below the
cooperative threshold (OMP=1, or `N < nthreads · 16`), the old
`esyrk_serial_blocked` reverted to per-jc-block beta-scale + scalar
`syrk_diag_add` + a recursive `egemm_` call for the trailing
rectangle. The egemm call allocated and freed ~2 MB Bp + ~256 KB Ap
*per jc-block* — at N=512 with nb=64 that's 8 mmap-heavy alloc/free
cycles per esyrk call.

Replaced with `esyrk_serial_inline`: same packers, same MR×NR kernel,
same `macro_kernel_tri` triangle-aware diagonal handling. One thread
walks (jc, pc, ic); each (ic, jc) tile classifies as `skip` /
`rect` / `tri` against the UPLO triangle. Buffers allocated once at
entry.

Perf @ N=512 OMP=1 kind10 (overlay GFs):

| key | before | after | delta |
|-----|--------|-------|-------|
| UN  | 2.45   | 3.04  | +24%  |
| UT  | 2.14   | 3.00  | +40%  |
| LN  | 2.39   | 2.98  | +24%  |
| LT  | 2.20   | 3.02  | +37%  |

@ N=128 (the worst old cells):
| UN  | 0.88   | 2.65  | +201% |
| LN  | 1.59   | 2.88  | +81%  |

This also wins for "OMP=N large but N<threshold" cases — e.g. OMP=12
N=128 LN went from 1.36 GFs to 2.68 GFs (the cooperative threshold
sends those to the serial inline path).

Bonus: removed the `extern egemm_(...)` declaration, the scalar
`syrk_diag_add` helper, and the `ESYRK_NB` env (replaced by `ESYRK_NC`
for NC-block sizing — defaults to 512, same as egemm).

### Rules

56. **Once a packed kernel exists, the serial path should use it
    too.** A "fall back to scalar diagonal + recursive gemm call"
    serial path leaves perf on the floor: scalar diag misses the
    packed kernel's stack-resident accumulators, and the recursive
    gemm call mmaps fresh Ap/Bp per block. Reuse the same packers
    and macro_kernel — only the threading goes away.

## Addendum 40: egemmtr — same packed-inline + tile-classify treatment (2026-05-21)

egemmtr had the same legacy structure as the old esyrk: per-jc-block
beta-scale + scalar diag_add + recursive `egemm_` call for the
trailing rectangle. The Add-38 sweep flagged it (median 2.45×) but
the sweep's OMP=1 cells were the *worst* outliers in the whole
catalog: N=64 had 8 cells in the 0.60–0.73× sub-parity range, the
worst being `UTT @ N=64` at 0.598×.

### Fix

`egemmtr.c` rewritten with the same packed-inline pattern as
`esyrk_serial_inline` (Add-39a) — packed `pack_A` / `pack_B`,
MR×NR `kernel_2x2` / `kernel_edge`, triangle-aware
`macro_kernel_tri`. The (jc, pc, ic) loop nest walks every tile;
each tile classifies against the UPLO triangle (skip / rect / tri).

Threading uses the egemm pattern (Add-34): one `omp parallel`,
shared Bp via `omp single` once per (jc, pc), private Ap per
thread, `omp for schedule(static, 1)` over ic so the triangular
load — early-ic threads see more skipped tiles for LOWER, more for
UPPER respectively — balances by interleaving.

Cooperative buffer-sharing (the SYRK trick) doesn't apply here
because A and B are independent matrices; each thread still has
to pack its own Ap.

### Result

Fuzz: 720/720 cases × {OMP=1, 4, 12} × {seed 1, 42, 1000} = 9 ·
80 = 720 cases clean.

OMP=1 — worst sub-parity cells healed:

| key  | N   | before | after |
|------|-----|--------|-------|
| UTT  | 64  | 0.598× | 1.195× |
| UNT  | 64  | 0.715× | 1.579× |
| LNT  | 64  | 0.717× | 1.604× |
| LNN  | 64  | 0.730× | 1.444× |
| UNN  | 64  | 0.732× | 1.050× |
| UNN  | 128 | 0.78×  | 1.633× |
| LNN  | 128 | 0.78×  | 1.628× |

All 32 OMP=1 cells now ≥ 0.95× vs migrated (LTN @ N=64 at 0.954
is the only one below 1.0×; the diagonal kernel is essentially
matched against migrated at this size). N=512 OMP=1 absolute
throughput went from ~2.5 GFs to ~3.0 GFs across all (UPLO, TA,
TB) cells.

OMP=4 reaches ~8 GFs at N=512 (~2.7× scaling), better than the
pre-Add-40 median of 2.45× but lower than egemm's ~3.5× since the
triangular work imbalance + tile-skip can't be perfectly
balanced under static scheduling. Acceptable — the OMP=1 win is
the structurally important one.

### Rule

57. **For triangular L3 routines like SYRK, GEMMTR, the packed-
    inline + tile-classify pattern is the canonical fix.** Walk
    (jc, pc, ic), classify each (ic, jc) tile as skip / rect /
    tri against the UPLO constraint, and dispatch to
    `macro_kernel_rect` or `macro_kernel_tri` (entry-by-entry
    UPLO check only for sub-tiles that actually straddle the
    diagonal). Don't fall back to a scalar diag + recursive-gemm-
    call structure — every such case eventually exhibits the
    same N=64 sub-parity cliff and the same per-block packing
    tax.
