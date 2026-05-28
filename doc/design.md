# Hand-written parallel BLAS overlay — design & phased plan

Date: 2026-05-13  
Branch: `epblas-parallel`

This document specifies the plan for a hand-written, thread-parallel
overlay BLAS for the extended-precision targets (`kind10`, `kind16`,
`multifloats`). The migrated `${LIB_PREFIX}blas` archives produced by
the migrator are correctness-baselined (`1125/1125` differential
precision tests against a Netlib quad reference) but **serial and
unblocked** — they are 1:1 type-rewrites of Netlib's reference BLAS.
This project layers cache-blocked, OpenMP-parallel kernels on top
without touching the migrator pipeline.

## 1. Strategy

Build a **separate** static archive per target — `lib{e,q,m}blas_parallel.a`
— that supplies hand-written replacements for a small set of
high-leverage routines. The migrated archive continues to provide
everything else.

The overlay archive `lib{e,q,m}blas_parallel.a` is **always built**
(no opt-in gate). What `PARALLEL_BLAS` (default ON) controls is
whether the public CMake target `${LIB_PREFIX}blas` is rewired into a
**composite INTERFACE** that links the overlay *before* the migrated
archive. Concretely:

| target name                       | type      | role                                                |
|-----------------------------------|-----------|-----------------------------------------------------|
| `${LIB_PREFIX}blas_serial`        | STATIC    | the migrated archive (was `${LIB_PREFIX}blas`)      |
| `${LIB_PREFIX}blas_parallel`      | STATIC    | the hand-written C overlay (this project)           |
| `${LIB_PREFIX}blas`               | INTERFACE | composite: links `_parallel` then `_serial`         |
| `${LIB_PREFIX}blas_migrated`      | STATIC    | `objcopy`-renamed `_serial` (symbols → `*_migrated_`); test-only |

Effect: every existing `target_link_libraries(... ${LIB_PREFIX}blas)`
— in `${LIB_PREFIX}lapack`, `${LIB_PREFIX}scalapack`,
`${LIB_PREFIX}mumps`, downstream apps, the installed package — picks
up overlay symbols automatically. Linker resolves the overlay's
`egemm_` first; everything not in the overlay falls through to the
serial archive. No per-consumer link-line edits anywhere.

When `PARALLEL_BLAS=OFF`, `${LIB_PREFIX}blas` remains a plain STATIC
target pointing at the migrated archive — current behavior unchanged.
The overlay archive still builds and is exercised by its own
consistency / fuzz / bench tests, but no downstream consumer links
it. This mode exists for A/B'ing end-to-end performance.

The overlay source lives at `src/epblas-parallel/<target>/` in the
project repo and is staged into the unified build tree by
`migrator stage`.

### Language

**C99-and-up, not Fortran.** Reasons:

- Better auto-vectorization where the type permits.
- Idiomatic `restrict`, alignment attributes, `__attribute__((hot))`.
- `#pragma omp parallel for` more directly than Fortran's `!$omp`.
- Easier control of packing buffer layout and prefetch.

Type mapping:

| target        | C type                  | notes                                |
|---------------|-------------------------|--------------------------------------|
| `kind10`      | `long double`           | x86-64 80-bit, ABI matches `REAL(KIND=10)` |
| `kind16`      | `__float128` (quadmath) | every op is a libquadmath call       |
| `multifloats` | multifloats C API       | upstream provides `float64x2_t` etc. |

### Fortran ↔ C boundary

Hand-written kernels are exported with the gfortran symbol mangling
that the migrated archive uses (lowercase + trailing `_`). The overlay
symbol *is* the BLAS entry point — no `iso_c_binding` shim.

- Scalar args by pointer (Fortran reference semantics).
- Arrays as raw pointers, column-major.
- Character args: pointer + hidden trailing `size_t` length per
  gfortran ABI. A common helper `static int parse_trans(const char *,
  size_t)` normalizes to `'N' | 'T' | 'C'`.

### Primary numerical oracle: the migrated archive

The hand-written parallel kernels are tested **against the migrated
BLAS at the same precision** (not against the Netlib quad reference).

- Same precision both sides → only divergence comes from reduction
  order (blocking, threading).
- **Tolerance: 10 ulp relative error** (user-set). Bit-exact across
  thread counts is explicitly NOT required.
- The existing `tests/blas/level{1,2,3}` precision suite (vs quad
  reference) is kept as a secondary regression check — runs
  automatically against whatever resolves `egemm_`, so flipping
  `PARALLEL_BLAS=ON` re-tests the overlay at the existing tolerance.

To allow both implementations to coexist in one test binary, the
migrated archive is renamed via `objcopy --redefine-syms` to
`*_migrated_` (`egemm_ → egemm_migrated_`, etc.). The test program
calls both `egemm_` (overlay) and `egemm_migrated_` (renamed migrated)
and diffs.

## 2. Routine priority

| phase   | routines                                           | rationale |
|---------|----------------------------------------------------|-----------|
| P2.a    | `egemm` / `ygemm`                                  | Only L3 routine that matters for downstream LAPACK/ScaLAPACK perf. Goto's blocked-and-packed algorithm. |
| P2.b    | `etrsm` / `ytrsm`                                  | Block-triangular solve, reduces to gemm on trailing block. |
| P2.c    | `esymm`/`yhemm`, `esyrk`/`yherk`, `esyr2k`/`yher2k`, `etrmm` | All reduce to gemm. |
| P3      | L2 `egemv` / `ygemv`                               | Memory-bound; OMP across rows, blocked for L2. |
| P4      | L1 `eaxpy` / `edot` / `enrm2`                      | Likely skip — OMP overhead exceeds arithmetic for typical sizes. Only if profile says so. |

Each phase only lands once fuzz + bench are green on the previous one.

## 3. Build integration

- Source tree: `src/epblas-parallel/<target>/{egemm.c, etrsm.c, ...}` plus
  `src/epblas-parallel/<target>/CMakeLists.txt` that defines the static
  archive `${LIB_PREFIX}blas_parallel`.
- Top-level `src/epblas-parallel/CMakeLists.txt` gates on
  `option(PARALLEL_BLAS)` and dispatches to the per-target subdir.
- The top-level `CMakeLists.txt` adds `add_subdirectory(src/epblas-parallel)`
  unconditionally when at least one target's per-precision package is
  discoverable on `CMAKE_PREFIX_PATH`.
- `migrator stage` copies `src/epblas-parallel/` into the staging dir
  (one-line addition to `cmd_stage` in `src/migrator/__main__.py`).
- Compile flags: `-O3 -fopenmp -march=native -ffp-contract=fast`
  (env-overridable per kernel for the bench autotune step).

## 4. Test infrastructure

Two test layers, both always-on whenever `PARALLEL_BLAS=ON`.

### 4.1 Consistency suite — **new, primary**

`tests/epblas-parallel/consistency/` — one driver per routine.

- Calls both `egemm_` (overlay) and `egemm_migrated_` (renamed migrated).
- Compares result matrices/vectors in `REAL(KIND=10)` (same precision
  both sides).
- Tolerance: `max_rel_err ≤ 10 * eps10` per element (kind10), with
  `eps10 ≈ 1.08e-19`. Equivalent thresholds for kind16, multifloats.
- Fails with the seed printed so failures are reproducible.

### 4.2 Precision regression — existing `tests/blas/level{1,2,3}`

Unchanged. Runs against whatever resolves the BLAS symbols. With
`PARALLEL_BLAS=ON` the overlay gets exercised; tolerance is the
existing `c * K * eps_quad` scheme. Catches any accuracy regression
the parallel kernel introduces relative to the Netlib quad oracle.

### 4.3 Fuzz harness — **new**

`tests/epblas-parallel/fuzz/`. Per routine, parameterized:

- Random `M, N, K` mixing edges {0, 1, 2, MR-1, MR, MR+1, 2·MR, ...}
  with mid (32–256) and tail (≥KC, ≥MC, log-distributed).
- Random `LDA, LDB, LDC ≥ required` (padding stresses stride code).
- All TRANS/UPLO/DIAG combos.
- Random `alpha, beta` including {0, 1, -1, denormal, near-overflow}.
- Random `INCX, INCY` ∈ {1, 2, -1, -3} for L1/L2.
- Seed from `BLAS_FUZZ_SEED` env (default: time-based, logged).
  Fail-with-seed for repro.
- Iteration count: `BLAS_FUZZ_CASES` (default 500 fast, 50000 nightly
  via `ctest -L nightly`).
- Oracle selectable via `BLAS_FUZZ_ORACLE = migrated|quad|both`
  (default `both`).

### 4.4 Threading torture

Fuzz re-run at `OMP_NUM_THREADS ∈ {1, 2, 4, 8, all}`. Pass if every
thread count is within 10 ulp of every other (not bit-exact).

## 5. Benchmark harness — **new**

`tests/epblas-parallel/bench/`. Per routine:

- Geometric size sweep (e.g. GEMM: 64, 128, 256, 512, 1024, 2048).
- Warm-up + N timed iterations; report median GFLOP/s, min, max.
- Block sizes `MC, KC, NC` taken from env (`EBLAS_MC`, etc.) so a
  Python launcher can grid-search.
- JSON output under `reports/`, schema mirroring
  `precision_reports/`.
- A `scripts/bench_report.py` aggregator that plots GFLOP/s vs N and
  speedup vs threads, and compares to the serial migrated baseline
  (`PARALLEL_BLAS=OFF` build).

## 6. Kernel design (kind10, GEMM)

Goto's blocked-and-packed scheme:

- Outer blocks: `NC` (~1024 cols of B/C), `KC` (~256), `MC` (~256).
  Tunable via env in bench.
- **Pack** A panel (MC×KC, into MR-row stripes) and B block (KC×NC,
  into NR-column stripes) into 64-byte-aligned scratch
  (`posix_memalign` / `aligned_alloc`), one per thread.
- **Micro-kernel** MR×NR = 4×4 for real, 4×2 for complex. Accumulator
  block kept in scalar registers (no SIMD intrinsics — gfortran/gcc
  have no 80-bit vector ISA on x86-64). `static inline`,
  `__attribute__((hot))`.
- `restrict`-qualify all inner pointers.
- **OpenMP**: `#pragma omp parallel for schedule(static)` on the
  NC-loop. Per-thread packing scratch indexed by
  `omp_get_thread_num()`. Reduction order is *not* preserved across
  thread counts (allowed by the 10-ulp tolerance).

For **kind16**: same skeleton; every inner FMA is a libquadmath call.
No vectorization. Blocking still pays for cache + TLB reasons.

For **multifloats**: inner kernel becomes an error-free transform
sequence. Consider exposing a fused `dot_acc` from upstream
multifloats if available. OMP outside as before.

## 7. Phasing

| step | deliverable                                                                            | gate |
|------|----------------------------------------------------------------------------------------|------|
| **0**  | C build plumbing: `src/epblas-parallel/` skeleton, `PARALLEL_BLAS` option, passthrough stub for `egemm_` | builds clean; existing 1125 unaffected when `PARALLEL_BLAS=OFF` |
| **0b** | `objcopy --redefine-syms` step that emits `lib${LIB_PREFIX}blas_migrated.a`; first test exe links both | passthrough call lands in migrated impl |
| **0c** | `cmd_stage` copies `src/epblas-parallel/` into staging                                   | `migrator stage` produces a self-contained tree |
| **0d** | Composite rewiring of `${LIB_PREFIX}blas` (rename existing target → `_serial`, add INTERFACE shim) under `PARALLEL_BLAS=ON` | `${LIB_PREFIX}lapack` links unchanged, downstream consumers transparently pick up the overlay |
| 1     | Fuzz harness for `egemm` (runs against the migrated archive only — validates the harness itself) | finds zero false positives on migrated-vs-migrated |
| 2     | Bench harness for `egemm` (baseline numbers committed)                                 | reproducible GFLOP/s for migrated baseline |
| 3     | Serial blocked + packed `egemm` in C (no OMP yet)                                      | fuzz green; bench within ~10% of serial Netlib at N≥256 (kind10 is compute-bound on x87, so blocking is a wash perf-wise but enables step 4 parallelization) |
| 4     | Add OMP to `egemm`                                                                     | fuzz green at all thread counts; ~linear scaling to N_core/2 |
| 5     | Block-size autotune sweep; commit tuned `MC/KC/NC`                                     | aggregator picks winners; merge to `develop` |
| 6     | Repeat for `ygemm` (complex)                                                           | fuzz + bench |
| 7     | `etrsm` / `etrmm` / `esymm` / `esyrk` family                                           | fuzz + bench per routine |
| 8     | Port to `kind16`                                                                       | reuse harness 1:1 |
| 9     | Port to `multifloats`                                                                  | reuse harness 1:1 |

Phases 1–4 stay on branch `epblas-parallel`; merge to `develop` only
after step 5 lands and is green.

## 8. Tolerance & determinism

- Per-test pass criterion: `max(|r_overlay - r_migrated| / |r_migrated|)
  ≤ 10 * eps_target` (with `eps_target = epsilon(1.0_K)`).
- Bit-exact across thread counts NOT required — frees the OMP
  partitioning to choose any schedule.
- Determinism within a fixed `OMP_NUM_THREADS` and fixed schedule is
  guaranteed by the algorithm (no atomics, no dynamic schedule for the
  reduction loop).

## 9. Open questions / explicit defaults

| item                  | default                                                           |
|-----------------------|-------------------------------------------------------------------|
| branch                | `epblas-parallel`; merge to `develop` after step 5                  |
| bench iterations      | 5 timed + 1 warm-up; fuzz fast=500, nightly=50000                 |
| determinism           | 10 ulp rel err; not bit-exact                                     |
| overlay packaging     | separate `lib*blas_parallel.a`; opt-in via `PARALLEL_BLAS=ON`     |
| Fortran ABI           | gfortran lowercase + trailing-`_` mangling, no `iso_c_binding` shim |
| symbol-aliasing trick | `objcopy --redefine-syms`; fallback `-Wl,--wrap` if it misbehaves |

## 10. Measured performance

Box: 4C/8T, gfortran-13 / gcc-13. `bench_*gemm` median of 2–3 timed
iterations after one warmup.

### Single-thread (`OMP_NUM_THREADS=1`) — cache sweep

Bench measures the square cube case `M = N = K = LDA = LDB = LDC = s`
(see `bench_gemm_real_body.fypp` — same `s` plugged into all six BLAS
shape args, no padding). Real LAPACK workloads include rectangular
panels with `K ≪ N`; perf there isn't covered by these numbers.

Box caches: per-core L1d=37 KB, L2=1.5 MB; shared L3=10 MB. Element
size is 16 B for all three precisions, so the per-matrix footprint
is `s² × 16 B`: s=512 → 4 MB ≈ L2 spill, s=1024 → 16 MB > L3.

Kernel choice: **kind10 and kind16 use the inner-product (DDOT)
micro-kernel**; **multifloats uses the outer-product (rank-1)
form**. The choice is precision-dependent — see "Kernel structure"
below.

#### kind10 (`long double`, x87) — inner-product

| s | overlay | migrated | overlay/migrated |
|---|---|---|---|
| 128  | 1.80 | 2.50 | 0.72× |
| 256  | 1.89 | 2.53 | 0.75× |
| 512  | 1.89 | 2.51 | 0.75× |
| 1024 | 2.17 | 1.32 | **1.65×** |
| 2048 | 2.21 | 1.30 | **1.71×** |

At s ≤ 512 the migrated impl wins — its triple loop fits in cache
and has zero blocking overhead. At s ≥ 1024 the working set spills
L2 → L3 → RAM, migrated drops to 1.3 GFLOP/s, and overlay's
blocking + register accumulator overtake at **1.7×**.

#### kind16 (`__float128`, libquadmath) — inner-product

| s | overlay | migrated | overlay/migrated |
|---|---|---|---|
| 128  | 0.060 | 0.054 | 1.10× |
| 256  | 0.060 | 0.056 | 1.06× |
| 512  | 0.059 | 0.058 | 1.02× |
| 1024 | 0.060 | 0.059 | 1.01× |

Cache effect still invisible (libquadmath function-call overhead
dominates), but inner-product's reduced C traffic and register
accumulator buy a small edge at every size — overlay slightly
ahead at all sizes single-threaded.

#### multifloats (DD, AVX2 4-wide SIMD by default; see §10.4) — was outer-product scalar

| s | overlay | migrated | overlay/migrated |
|---|---|---|---|
| 128  | 0.500 | 0.123 | 4.08× |
| 256  | 0.509 | 0.123 | 4.14× |
| 512  | 0.507 | 0.122 | 4.16× |
| 1024 | 0.505 | 0.123 | 4.12× |
| 2048 | 0.506 | 0.123 | 4.12× |

Also flat across sizes — but at a constant ~4.1× advantage. Neither
impl is memory-bound; the gap is purely inlining (the C++ kernel
folds `multifloats::float64x2::operator*` into the inner loop) vs
gfortran's elemental-wrapper boundary on every element. Cache
doesn't matter when the inner op is a few inlined doubles.

**Takeaway**: cache blocking pays only when arithmetic is fast
enough to expose memory traffic. On this box kind10 hits that
crossover at s≈1024; kind16 and multifloats stay arithmetic-bound
across the measured range.

#### Single-thread per-transpose (kind10)

Same column-major cache effect that shows up at OMP=4 is already
visible serially — confirms it's an algorithmic / access-pattern
fact, not a parallelism artifact.

| trans | overlay s=1024 | migrated s=1024 | overlay s=256 | migrated s=256 |
|---|---|---|---|---|
| NN    | 2.18 | 1.25 | 2.20 | 1.34 |
| NT    | 2.21 | 1.23 | 2.19 | 1.31 |
| NC    | 2.23 | 1.23 | 2.22 | 1.32 |
| **TN** | 2.17 | **1.61** | 2.22 | **2.40** |
| TT    | 2.24 | 0.63 | 2.24 | 1.27 |
| TC    | 2.19 | 0.62 | 2.21 | 1.13 |
| **CN** | 2.13 | **1.53** | 2.26 | **2.41** |
| CT    | 2.24 | 0.61 | 2.15 | 1.31 |
| CC    | 2.21 | 0.61 | 2.24 | 1.30 |

- Overlay tracks ~2.2 GFLOP/s everywhere.
- At s=256, migrated TN/CN **beat the overlay** (2.40 vs 2.22 →
  0.92×) — Netlib's natural loop nest hits two-contiguous-columns
  dot products with zero blocking overhead, and the problem fits in
  cache. Overlay catches up at s=1024 once migrated starts spilling.
- At s=1024, even the migrated TN/CN best case (1.6) lags the
  overlay (2.2); the worst case (TT/TC/CT/CC) is 3.6× behind.

#### Single-thread per-transpose (kind16, multifloats)

- **kind16** OMP=1, s=512: both impls flat 0.058–0.060 across all
  9 combos. Overlay 0.93×–1.10× per combo. Transpose dispatch is
  invisible behind the libquadmath call cost.
- **multifloats** OMP=1, all sizes: overlay flat 0.50, migrated flat
  0.12, every combo. Uniform 4.0–4.3× speedup. Inlined DD ops
  dominate; memory shape is irrelevant.

Per-trans is only an interesting story on kind10 — the one
precision where compute is fast enough for memory access patterns
to show up.

### Kernel structure (outer-product vs inner-product)

After packing, the inner kernel has two natural shapes:

| variant | layout of Ap | inner loop | C traffic | accumulator |
|---|---|---|---|---|
| outer-product (rank-1) | col-major | `cj[i] += t · ap[i]`, `t` hoisted from Bp | one read+write per (i, j, p) | none — accumulates straight into memory |
| inner-product (DDOT)  | row-major | `sum += ai[p] · bj[p]`, accumulator scalar | one read+write per (i, j) | scalar accumulator stays in register across p |

The trade-off:
- Outer-product gives the compiler many independent `cj[i]`
  store streams to pipeline → high instruction-level parallelism,
  but `pb` more C writes than the inner-product form.
- Inner-product has a tight data dependency through the accumulator
  → no ILP on the contraction, but the accumulator never touches
  memory and C is touched once per (i, j).

Empirically on this box:

| target | outer-product OMP=1 | inner-product OMP=1 | choice |
|---|---|---|---|
| kind10 @ s=2048 | 1.35 GFLOP/s | 2.21 | **inner-product** (1.6× faster) |
| kind16 @ s=1024 | 0.058 | 0.060 | **inner-product** (small win) |
| multifloats @ s=512 | 0.51 | 0.33 | **outer-product** (1.5× faster) |

- kind10's bottleneck at large s is **memory traffic** (long-double
  C writes touch a 16 MB buffer); halving C traffic helps.
- multifloats's bottleneck is **the DD accumulator dependency
  chain** — each `sum += ai[p] * bj[p]` is ~12 inlined hardware FMAs
  and the next iter waits on the accumulator. Outer-product gives
  the compiler `ib` independent C-streams to pipeline, lifting ILP.
- kind16 is in between, with inner-product slightly ahead because
  libquadmath calls dominate everything else.

So the codebase currently mixes kernel shapes per target. Both
shapes live as drop-in replacements of the same `inner_kernel` (and
its matching `pack_A` layout) inside `egemm.c` / `qgemm.c` /
`mgemm.cpp` respectively. The choice is one local edit per target;
no shared infra changes.

### AVX2 SIMD micro-kernel for multifloats — **default ON**

Double-double's 16-byte type fits naturally into ymm registers (4×
double per ymm = 4 parallel DD values). The error-free transforms
(`twoprod`, `twosum`) at the heart of DD arithmetic are
embarrassingly parallel across lanes — no horizontal reductions,
no cross-lane dependencies.

Implementation in `src/epblas-parallel/multifloats/mgemm_simd_kernel.h`:

  twoprod(a, b) → (p, e):  p = a*b, e = fma(a, b, -p)   /* exact */
  twosum(a, b)  → (s, e):  6-op variant
  dd_mul(ah, al, bh, bl) → (rh, rl)
  dd_add(ah, al, bh, bl) → (rh, rl)

All using `_mm256_*pd` on AVX2 + FMA. Micro-kernel shape MR=1,
NR=4 (1 row of A × 4 cols of B per iteration). B is repacked in
SoA layout (separate `hi[]` / `lo[]` arrays) so the inner-p loop
loads 4 contiguous doubles per iteration with `vmovupd`. A stays
AoS scalar (broadcast per-p with `vbroadcastsd`).

Gated on `MBLAS_SIMD_DD` (default ON when the compiler accepts
`-mavx2 -mfma`, which is the case for any halfway-recent x86_64).

Micro-kernel sizing: MR is the number of A rows processed per
micro-kernel call (NR=4 is fixed by the AVX2 lane count). Each row
runs an independent dd_mul → dd_add chain that shares the same
`vmovupd` of B; multiple chains expose ILP to hide the EFT
dependency latency. Picked via the `MBLAS_SIMD_MR` cmake cache
variable, default **3** (kernel is templated on MR — see
`inner_kernel_simd_mr<int MR>` in mgemm.cpp).

Empirical sweep on Raptor Lake (i3-1315U), OMP=1, s=1024, NN.
Kernel is templated on `(MR, NR_PAN)` where `NR_PAN` is the number
of stacked 4-lane ymm panels along the j-axis (so NR_PAN=1 → 4
cols/call, NR_PAN=2 → 8 cols/call).

| MR | NR_PAN | parallel chains | GFLOP/s |
|---|---|---|---|
| 1 | 1 | 1 | 1.89 |
| 2 | 1 | 2 | 3.50 |
| 3 | 1 | 3 | 3.70 |
| 4 | 1 | 4* | 3.78 (\*spills slightly) |
| 1 | 2 | 2 | 3.45 |
| 2 | 2 | 4 | 3.69 |

The relevant quantity is **MR × NR_PAN = number of independent
dd_mul → dd_add chains in flight**. Going 1 → 2 chains roughly
doubles throughput; past 3 chains the FMA unit is saturated and
the ceiling is ~3.7 GFLOP/s regardless of how you spend the
parallelism budget. (MR=2,NR_PAN=2) and (MR=3,NR_PAN=1) tie.

So **wider NR is not a new lever** — just a different way to spend
the same MR × NR_PAN budget. The amortization-of-loads benefit
(fewer pack_B / vmovupd per output cell) doesn't show up because
loads weren't the bottleneck; the FMA throughput is.

Default kept at **MR=3, NR_PAN=1** (simplest config that reaches
the FMA ceiling). At MR=4 the 8 ymm accumulators + 8 broadcasts +
2 B + 2 scratch saturate the 16 ymm registers and gcc starts
spilling. Below MR×NR_PAN=2 the EFT chain serializes — single chain
in flight, FMA latency unhidden.

Trailing odd-modulo rows (`ib` not divisible by MR) handled by a
separate MR=1 tail loop after the main MR=`MGEMM_SIMD_MR` loop —
keeps the main loop branch-free.

Measured on Raptor Lake (i3-1315U, AVX2 + FMA, no AVX-512):

| | OMP=1 | OMP=4 (s=1024) |
|---|---|---|
| **scalar outer-product baseline** | 0.51 GFLOP/s | (≈1.26) |
| **AVX2 SIMD MR=1** | 1.89 | 4.70 |
| **AVX2 SIMD MR=2** | 3.49 | 7.58 |
| **AVX2 SIMD MR=3 (default)** | **3.74** | **7.71** |
| **MR=3 vs scalar overlay** | **7.4×** | **6.1×** |
| **MR=3 vs migrated** | **30×** | **63×** |

Per-trans variation under 5% (DD ops dominate everything; even the
non-NN packing's strided read is invisible behind the EFT chain).

The 63× over migrated is the largest speedup of any kernel in this
overlay — the migrated Fortran goes through an elemental wrapper
around each scalar DD op (call overhead) and uses no SIMD; the
overlay fixes both at once.

### AVX2 SIMD wgemm (complex DD)

Complex DD multiplication = 4 dd_muls + 1 dd_sub + 3 dd_adds per
element. The 4 dd_muls in `(a+bi)·(c+di) = (ac−bd) + (ad+bc)i` are
**independent within a single complex multiplication** — i.e. the
inner kernel already has 4-way intra-cell ILP before we add any
outer MR / NR_PAN parallelism. So the FMA throughput ceiling is
reached at the smallest tile.

SoA layout: 4 separate arrays per packed B panel (re_hi, re_lo,
im_hi, im_lo). Conjugate-transpose 'C' folds into pack via
negating im_h / im_l during the copy.

Sweep, Raptor Lake i3-1315U, OMP=1 NN s=512 (complex GFLOP/s
= 8·s³/t):

| MR | NR_PAN | chains | GFLOP/s |
|---|---|---|---|
| 1 | 1 | 4† | 3.87 |
| 2 | 1 | 8† | 3.86 |
| 1 | 2 | 8† | 3.87 |
| 2 | 2 | 16† | 3.94 |

(† intra-cell parallelism is 4 dd_muls; total chains = 4 · MR · NR_PAN.)

The default (MR=1, NR_PAN=1) is already at the ceiling, ~99% of the
maximum reached anywhere. Wider configurations give 2% more for
2-4× the register pressure — not worth the complexity.

Headline at default:

| | OMP=1 NN s=1024 | OMP=4 NN s=1024 |
|---|---|---|
| migrated | 0.22 GFLOP/s | 0.22 |
| **AVX2 SIMD overlay** | ~3.85 | **8.44** |
| vs migrated | ~18× | **39×** |

Worst-case for the migrated path is CC (conjugate-transpose both
sides): 0.15 GFLOP/s vs overlay 8.06 → **55×**.

### Register-tiled (MR × NR) micro-kernel — tried, abandoned

The next natural step after outer-product / inner-product is the
BLIS-style register-tile: keep an `MR × NR` block of C accumulators
in registers across the entire `pb` contraction, with one read+write
of C per tile (not per (i, j)). Tried `MR = 2, NR = 4` for
multifloats (8 DD accumulators = 16 doubles = 8 SSE regs, well
within the 16 xmm reg budget).

Head-to-head OMP=4 on this box (overlay GFLOP/s):

| s | outer-product | register-tile (2×4) |
|---|---|---|
| 256  | 0.50 | 0.49 |
| 512  | 0.96 | 0.90 |
| 1024 | 1.23 | 1.24 |
| 2048 | 1.26 | 1.25 |

**No improvement.** The outer-product form already has `ib`-way ILP
(64 independent `cj[i]` streams), so the register tile's MR · NR = 8
parallel accumulators is a strict subset. The actual bottleneck on
this precision is the **inlined DD FMA chain depth** — each
`muldd / adddd` is ~12 hardware ops with internal latency that the
register tile can't shorten. Both packers and the micro-kernel cost
more code complexity for zero gain.

Skipped for kind10 / kind16: x87 has only 8 stack slots so a 4×4
register tile spills immediately on kind10; libquadmath function
calls dominate everything on kind16 so the structural change is
irrelevant.

### Inlining `__multf3` / `__addtf3` — tried, abandoned

These are libgcc's soft-float quad-precision multiply and add — what
gcc emits for every `__float128` `*` and `+` operator. They live in
libgcc as external symbols, called via PLT trampoline. Hypothesis:
inlining the bodies into the hot loop would save the call overhead.

Vendored gcc's `libgcc/soft-fp/` template headers under
`external/libgcc-softfp/`. Wrote `static inline
__attribute__((always_inline))` wrappers `qmul`, `qadd`, `qsub` in
`src/epblas-parallel/kind16/qmath_inline.h`. Gated on
`-DQBLAS_INLINE_SOFTFP=ON`. Verified the expansion lands in the
binary: `qgemm_._omp_fn.0` body grew from ~250 to ~5000
instructions (correct: each `qmul` is ~700 insns, each `qadd` is
~1200).

OMP=4, NN, s=512 head-to-head:

| build | GFLOP/s |
|---|---|
| call-based (default) | 0.118 |
| inline soft-fp        | 0.116 |

**No speedup.** The PLT call overhead (~5–10 cycles) is < 5% of the
~200–400-cycle soft-float body. Inlining saves that, but the inner
loop's instruction footprint balloons and L1i fetch latency eats
back whatever was gained.

The infrastructure (vendored headers, inline shim, cmake flag) is
kept in place — off by default — so the experiment stays
reproducible and the negative result is documented in code. If a
future toolchain version closes the gap (e.g. better register
allocation in the inlined bodies, or LTO across libgcc), flipping
the flag is one cmake edit.

### `fmaq()` — tried, much *worse*

The natural next thought is libquadmath's `fmaq(a, b, c) = a*b + c`:
one function call instead of two (`__multf3` + `__addtf3`) and a
single rounding, so should be at least as fast.

Replaced `sum += ai[p] * bj[p]` with `sum = fmaq(ai[p], bj[p], sum)`
(gated on `-DQBLAS_USE_FMAQ=ON`). Bench OMP=1:

| build | GFLOP/s |
|---|---|
| call-based (default, two soft-fp ops) | 0.058 |
| fmaq                                  | 0.0034 |

**~17× slower.** The reason is the inverse of the hardware story.
In hardware FMA is faster than separate ops because the multiplier
emits the full double-width intermediate and FMA reuses it
directly. In software, "single rounding" requires computing the
**exact 226-bit product** of two 113-bit mantissas, aligning it
with the 113-bit summand, summing (possibly with massive
cancellation), and rounding once. The separate-ops path rounds at
113 bits between mul and add and never materializes the full
intermediate — much less work.

So IEEE-correct soft-float FMA is more accurate but vastly
slower. Worth noting because the intuition that "1 call < 2 calls"
fails completely here.

(Infrastructure kept; gated off by default. Flag: `QBLAS_USE_FMAQ`.)

### SIMD-vectorized soft-fp — analyzed, infeasible on AVX2-only

Would 4-wide (AVX2) or 8-wide (AVX-512) parallel `__float128`
multiplies via SoA limb layout pay off? The throughput bound:

The hot operation is a 113×113 → 226-bit mantissa multiply. Scalar:
4 `mulq` instructions = ~4–6 cycles end-to-end (mulq is
1 op/cycle, native 64×64 → 128). On **AVX2 the only widening
multiply is `vpmuludq` (32×32 → 64)**, which needs ~8 instructions
to synthesize one 64×64 → 128 per lane:

| ISA | per-fraction-multiply throughput vs scalar |
|---|---|
| AVX2 only  | **0.5× — slower** (16 cycles SIMD vs 4 cycles scalar for 4-wide) |
| AVX-512 + IFMA52 (Ice Lake-X / Sapphire Rapids / newer Xeon) | ~7× faster (8-wide `vpmadd52*uq` with native 52×52 widening) |

The scalar `mulq` is too good a competitor on AVX2 — the SIMD path
adds carry-handling shifts/adds that erase the parallelism gain.
Existing SIMD `__float128` libraries (Sleef, Intel IPP-crypto)
exclusively target the AVX-512 IFMA52 path.

The box this analysis was done on is an Intel i3-1315U (13th gen
Raptor Lake): AVX2 + FMA, no AVX-512 (Intel disabled it on
consumer Alder/Raptor Lake). So SIMD soft-fp is a net loss here.

### Conclusion — kind16 perf wall-bounded by hardware

All five potential paths analyzed; none survives:

| option | outcome |
|---|---|
| #1 pack/unpack elision | inline-softfp showed zero gain → not the bottleneck |
| #2 unpack Bp once into SoA | est. 3–5 %, not worth the engineering |
| #3 skip case classification | est. 5 %, gcc may already do it via branch prediction |
| #4 SIMD-vectorize soft-fp | needs AVX-512 IFMA52, absent on this box |
| #5 loop-level exponent management | dropped by user direction |

So kind16 perf is **truly bound by hardware**. The single concrete
headroom direction is upgrading to a CPU with native `_Float128`
operations — Sapphire Rapids (Xeon server, mid-2023+) ships them,
gcc emits them automatically when `-march=` enables the relevant
ISA. On consumer Raptor/Alder Lake / older Skylake-X, kind16 is at
the floor.

Worth nailing down what is *not* a path: replacing `__float128`
with multifloats DD is sometimes proposed as a "faster quad", but
**DD has double's exponent range** (~10⁻³⁰⁸ … 10³⁰⁸, 11-bit
exponent) while `__float128` carries a 15-bit exponent (~10⁻⁴⁹³²
… 10⁴⁹³²). Any code that takes magnitudes outside double's range
— which is exactly what users reach for extended precision
*for* — would silently overflow/underflow under DD where
`__float128` rounds normally. DD is a different precision, not a
faster quad.

Conclusion: register tiling is the right answer for IEEE
double-precision GEMM on hardware with abundant SIMD registers and
hardware FMA throughput high enough to expose memory traffic. None
of our extended-precision targets sit there; we leave the kernel
shape at outer- / inner-product as appropriate per target.

Reading:
- **kind10 / kind16**: overlay roughly ties the migrated archive
  single-threaded. Both precisions are arithmetic-bound on this
  hardware (x87 80-bit on a shared FPU; libquadmath function calls
  per op). Cache blocking + packing don't recover cycles the FPU /
  libquadmath doesn't have. Overlay's value here is the OMP scaling
  the migrated Fortran version can't access.
- **multifloats**: overlay **4× faster at OMP=1**. Two stacked
  effects: (a) the C++ kernel inlines `multifloats::float64x2`'s
  overloaded `operator*` / `operator+` (and their underlying
  error-free transforms) directly in the hot loop, whereas the
  migrated Fortran archive routes every DD op through gfortran's
  elemental wrappers — one function-call boundary per element;
  (b) cache blocking + packing pays for a 16-byte scalar type at
  typical L1/L2 footprints.

### Parallel scaling (`OMP_NUM_THREADS=4`, s=1024 unless noted)

#### kind10 — per-transpose breakdown

The migrated kernel's perf varies 3× across the 9 transpose combos
because column-major access patterns hit different cache profiles
(textbook "A^T·B is cache-friendliest" effect — TN/CN read two
contiguous columns in the inner loop; TT/TC/CT/CC have the worst
stride pattern). The overlay packs both operands into a uniform
layout, so its perf is essentially constant — packing turns a 3×
swing into noise.

| trans | overlay | migrated | overlay / migrated |
|---|---|---|---|
| NN | 7.70 | 1.27 | 6.1× |
| NT | 7.57 | 1.23 | 6.2× |
| NC | 7.42 | 1.24 | 6.0× |
| TN | 7.54 | **1.56** | 4.8× |
| TT | 7.56 | **0.55** | **13.7×** |
| TC | 7.53 | 0.61 | 12.4× |
| CN | 7.36 | **1.64** | 4.5× |
| CT | 7.29 | 0.63 | 11.5× |
| CC | 7.47 | 0.63 | 11.9× |

So packing's biggest practical win on kind10 isn't the headline
GFLOP/s — it's **uniformity**. Downstream LAPACK calls
`DGEMM('T','N',...)` and `DGEMM('N','T',...)` in roughly equal
proportions; performance that doesn't depend on which one you
called is real engineering value.

#### kind16, multifloats — uniform across combos

For these precisions, arithmetic cost (libquadmath calls / inlined
DD ops) dominates everything else so the transpose dispatch doesn't
swing perf — both impls track flat across the 9 combos. Overlay
wins 1.9–2× (kind16) and 7–10× (multifloats) per combo, with
≤10% variation by transpose.

## 11. References

- Project memory: [project-epblas-parallel-design](../../../.claude/projects/-home-kyungminlee-Code-fortran-migrator/memory/project_parallel_blas_design.md)
- Goto, K., van de Geijn, R. — *Anatomy of High-Performance Matrix
  Multiplication* (ACM TOMS, 2008). Source of the
  MC/KC/NC + pack-A / pack-B layout.
- BLIS framework — practical reference implementation of the same
  algorithm, in C, for arbitrary scalar types.

(Inadvertent section numbering: §11 above is the references; §10
inserted with measured perf.)
