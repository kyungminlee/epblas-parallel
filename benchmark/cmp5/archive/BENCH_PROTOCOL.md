# gap5 benchmark protocol

How `epblas-parallel` measures par-vs-ob timing so the numbers are trustworthy.
Scripts live here (`reports/cmp5/`); raw sweeps + the scoreboard land in the
gitignored scratch tree `workspace/files/gap5/`.

## The one thing that matters: systematic error, not random noise

Every false verdict this project has hit — qsyrk "1.14 serial gap", ieamax /
eynrm2 "serial gaps", phantom OMP4 cells — was a **systematic** artifact (a
different CPU-frequency state, block placement, or contention between two
processes), not random scatter. We compare a **ratio** (par/ob), so the protocol
exists to make that ratio robust to the dominant systematic: CPU frequency.

On the bench box (i7-8700, base 3.2 GHz, turbo 4.6 GHz, `powersave` governor) a
single cell's absolute ns swings ~40% purely from turbo/DVFS. Absolute ns is
therefore only meaningful at a pinned frequency; the **ratio** is what we trust.

## Two defenses against frequency error

1. **Pin it (when root is available).** `run_gap5.sh` best-effort sets every
   CPU's governor to `performance` and disables turbo (`intel_pstate/no_turbo`
   or `cpufreq/boost`), restoring on exit. When the sysfs knobs aren't writable
   (the usual unprivileged case) it records `freq_pinned=0` in the `.meta`
   sidecar and proceeds — it does **not** fail. To pin manually, once:

       ! sudo cpupower frequency-set -g performance
       ! echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

2. **Cancel it by pairing (always).** Both arms can't live in one process — the
   two libraries export the same BLAS symbols, so linking them together
   mis-resolves shared helpers (segfaults/bogus maxerr). Cross-process is
   structural, so the finest pairing available is **per rep**: each binary
   invocation emits one number per cell, and we pair par-rep-`i` with
   ob-rep-`i`. Within a rep the two arms we compare run **back-to-back**
   (`run_gap5.sh` leg order `par1, ob1, par4, ob4`) at ~the same frequency, so
   frequency **cancels in the per-rep ratio**. The statistic is then taken on
   those ratios — this is the unprivileged substitute for a hardware pin.

Plus: pin threads (`taskset` omp1→cpu0, omp4→cpu0-3); never run two pinned
sweeps at once (contention poisons both); a ~0.4 s warmup spin ramps turbo to
steady state before the first timed call.

## Statistics (`agg_gap5.py`)

Per `(routine, key, size)` cell, from the per-leg rep series:

- **Point estimate — ratio-of-mins**: `min_rep(par) / min_rep(ob)`. The minimum
  over reps is the maximum-likelihood floor under one-sided additive noise
  (interrupts/eviction/DVFS only ever *add* time), so it beats mean/median for a
  single arm's deterministic cost.
- **Robust estimate — median of per-rep paired ratios + bootstrap 95% CI**.
  Pairing cancels shared-mode frequency drift; the median is outlier-robust; the
  bootstrap CI gives a real uncertainty band.
- **Significance flag**: a cell is `SLOW` only when the per-rep ratio's **CI
  lower bound > 1.0** (reproducibly slower than ob), replacing the old fixed
  `1.03` fudge. Magnitude (the median ratio) and significance (the CI) are
  separate axes — rank by magnitude, gate by significance.
- **`~DRIFT`**: ratio-of-mins and median-per-rep-ratio disagree by >0.04 →
  residual frequency drift or bimodality; the cell is contaminated, don't trust
  the absolutes (this is what catches the cross-process artifacts).

## Inputs and counters

- Bounded, deterministic fill (`perf_common.h`): unbounded `x := A·x` overflows
  → NaN/denormal → x87 ~100× slowdown that masquerades as a gap.
- Wall clock only (`CLOCK_MONOTONIC`); never `cycles:u` across thread counts
  (frequency-variant).
- `subject_ns` per invocation is currently the **mean** over `BLAS_PERF_ITERS`.
  The outer min-over-reps compensates (min-of-batch-means), but a per-invocation
  **min-of-batches** reduction in `perf_common.h` would be a strictly better
  floor — deferred because it shifts the absolute ns in every historical
  `*_interleaved.md` report (cross-cutting; needs explicit sign-off).

## Files

- `run_gap5.sh` — the sweep. Env: `ROUTINES` (required), `REPS` (default 11),
  `RAW`/`LOG`/`OUT_DIR`, `SKIP_BUILD=1`. Rebuilds `perf_$r`/`ep_perf_$r` before
  timing (stale-archive trap). Emits `…_raw.tsv` (+ `rep` column) and `…_raw.meta`.
- `agg_gap5.py RAW.tsv` — per-cell stats table, worst-ranked. Importable
  (`load`, `cell_stats`, `cell_flags`).
- `scoreboard.py [OUT.csv RAW…]` — consolidates the current post-fix raw files
  (newest-fix-wins per routine) into one ranked `workspace/files/gap5/scoreboard.csv`.
