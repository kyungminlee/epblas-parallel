# Dual-link benchmark protocol

How `epblas-parallel` measures par-vs-ob-vs-mig timing so the numbers are
trustworthy. The harness lives here (`bench/dual/`); generated drivers and raw
sweeps land in the gitignored scratch tree `workspace/files/gap5/nsbench/`.

This **supersedes** the cross-process `cmp5`/`gap5` harness, whose historical
reports are frozen under `bench/cmp5/archive/`. Do not cite those for current
verdicts.

## The one thing that matters: systematic error, not random noise

Every false verdict this project has hit — qsyrk "1.14 serial gap", ieamax /
eynrm2 "serial gaps", phantom OMP4 cells — was a **systematic** artifact (a
different CPU-frequency state, block placement, or contention between two
*processes*), not random scatter. We compare a **ratio** (par/ob, par/mig), so
the protocol exists to make that ratio robust to the dominant systematic: CPU
frequency and cross-process layout.

## The defense: one process, interleaved per rep

The old harness ran each arm in a **separate process** because the par/ob/mig
libraries all export the same BLAS symbols — link them together naïvely and the
shared helpers mis-resolve (segfaults / bogus maxerr). That separation was the
single largest source of systematic error: two processes see different page
placement, different turbo/DVFS states, and contend if run concurrently.

The dual-link harness removes the separation instead of defending against it:

1. **Namespace the archives** (`nsbuild.sh`). Each static archive is
   `objcopy --redefine-syms`'d to a per-leg prefix — `par_`, `ob_`, `mig_` — so
   all three implementations coexist in **one binary** with no symbol collision.
   The real linked routine is preserved: dispatch, thresholds, PLT, and the
   threaded path all run exactly as shipped.

2. **Interleave per rep on shared buffers** (`scripts/_perf_harness/dual.py`).
   For each rep the driver times the reset cost, then the `ob`, `par`, and `mig`
   legs **back-to-back on the same input buffers**, takes the min over reps per
   leg, and subtracts the reset floor. Because the legs we compare execute
   microseconds apart in the same process at the same frequency on the same
   pages, frequency and layout **cancel in the ratio** — structurally, not
   statistically.

Plus the standing discipline: pin threads (`taskset`, omp1→one core, omp4→four
cores); **never run two pinned sweeps at once** (contention poisons both —
verify the box is idle before timing); a warmup spin ramps turbo to steady
state before the first timed call.

## The bars (these OVERRIDE the generic "par ≤ ob" default)

Per `(routine, key, N)` cell, from the per-leg min-over-reps wall time:

- **Serial pass**: `par1 ≤ min(ob1, mig1)`. par must beat **both** the OpenBLAS
  clone **and** the gfortran-netlib migration, whichever is faster — `mig` is
  often the binding/fastest serial leg (e.g. Transpose stride-1 kernels). `par ≤
  ob` alone is **not** passing.
- **OMP=4 pass**: `par4 ≤ ob4`.
- **Scaling**: `par4 / par1` (smaller = faster; the threading headroom).

`agg_dual.py` prints the full table plus worst-N boards for each bar and a
`TOTAL FAIL` count (default bar 1.05).

## Reps discipline

- **Sub-2% verdicts need `REPS ≥ 40`.** min-of-10 fabricates phantom 1–2% gaps;
  a "leaner-codegen-yet-slower" floor is only real if it survives reps=40.
- The min-over-reps is the maximum-likelihood floor under one-sided additive
  noise (interrupts / eviction / DVFS only ever *add* time), so it beats
  mean/median for a single arm's deterministic cost.

## Inputs and counters

- **Bounded, deterministic fill.** Unbounded `x := A·x` overflows → NaN/denormal
  → x87 (and fp128) ~100× slowdown that masquerades as a gap. Solve/triangular
  fills force diagonal dominance so the result stays finite.
- **Wall clock only** (`CLOCK_MONOTONIC`); never `cycles:u` across thread counts
  (frequency-variant).
- **Report bare wall time (ns/call) + the par/ob (or par/mig) wall-time ratio,
  smaller = faster. Never convert to GF/s. Label the direction every time.**

## Files

- `nsbuild.sh <e|q|m>` — namespace the par/ob/mig archives into
  `lib_{par,ob,mig}_ns.a` (in the scratch tree).
- `run_dual.sh <e|q|m> [routine,...]` — end-to-end: build the CMake archives
  (anti-stale-archive), namespace them, generate one dual driver per routine
  (`scripts/gen_dual_harnesses.py`), compile each against the ns archives +
  refblas (`lsame_`/`xerbla_`), run OMP=1 and OMP=4 at `REPS`, then aggregate.
  Env: `REPS` (40), `CORE1` (2), `CORE4` (2-5), `OUT`, `SKIP_BUILD`, `NOAGG`.
- `agg_dual.py [results_dir] [topN] [bar]` — the scoreboard. Globs
  `*.omp{1,4}.txt`, keys by `(routine, key, N)`, applies the bars above.
- `scripts/gen_dual_harnesses.py [--family e|q|m] [--routines a,b] [--list]` —
  emit the dual drivers (also driveable standalone).

The drivers and all raw `.omp{1,4}.txt` data stay in
`workspace/files/gap5/nsbench/` (gitignored); never under `bench/`.
