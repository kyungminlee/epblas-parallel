# Benchmark

The overlay exists for speed, so a change to a hot path is not done until it is
re-timed. Performance is measured with the **in-process dual-link harness**:
the three implementations of each routine — `par` (our overlay), `ob` (the
OpenBLAS-derived reference), and `mig` (the gfortran netlib migration) — are
compiled into one binary with their symbols namespaced (`par_`/`ob_`/`mig_`) and
timed **interleaved per rep on the same buffers**. Because the legs run
microseconds apart at the same CPU frequency on the same pages, the `par/ref`
ratio is robust to the systematic error (turbo/DVFS, page placement) that
poisoned the old cross-process harness. We report **bare wall time (ns/call)**
and the ratio **par / reference, smaller = faster** — never GF/s.

## The two docs that own the detail

| Doc | Role |
|---|---|
| [`../../benchmark/dual/README.md`](../../benchmark/dual/README.md) | the **runbook** — prerequisites, the three entry points, env knobs, reading the board, machine hygiene, troubleshooting. |
| [`../../benchmark/dual/BENCH_PROTOCOL.md`](../../benchmark/dual/BENCH_PROTOCOL.md) | the **rationale** — why one process / interleaved / min-over-reps, and the pass/fail bars. |

They live beside the scripts (`benchmark/dual/`) so the harness and its docs stay in
sync. This page is the pointer from the dev-doc set; read the runbook first.

## The pass/fail bars

Per `(routine, key, N)` cell, from the per-leg min-over-reps wall time:

- **Serial** — `par1 ≤ min(ob1, mig1)`. par must beat **both** the OpenBLAS
  clone and the netlib migration, whichever is faster; `mig` is often the
  binding leg on stride-1 Transpose kernels, so `par ≤ ob` alone is **not** a
  serial pass.
- **OMP=4** — `par4 ≤ ob4`.
- **Scaling** — `par4 / par1` (report only; `< 1` = threading helps).

Cells are flagged at **par/ref > 1.02**; the reps≥40 harness is trustworthy to
sub-2%, so **1.00–1.02 is the noise band** — don't chase it.

## Running it

```bash
benchmark/dual/update_routine.sh e etbsv       # re-time one routine + refresh the .md
benchmark/dual/run_dual.sh q qsyrk,qsyr2k      # sweep a family → console board
cmake --workflow --preset sweep            # CI-style: run_dual.sh per family via ctest
```

`update_routine.sh` re-renders the committed `benchmark/results.md`; raw
data is heavy and gitignored under `workspace/files/gap5/nsbench/`. Commit the
updated scoreboard alongside the kernel change. Everything else — REPS≥40 for
sub-2% verdicts, one pinned sweep at a time, idle box — is in the runbook.

> **Machine hygiene, in one line:** never run two pinned sweeps at once and
> verify the box is idle before timing — contention fabricates gaps. See the
> runbook's "Machine hygiene" section.
