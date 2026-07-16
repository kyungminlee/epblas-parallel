# Running perf: the dual-link harness

A developer runbook for measuring `epblas-parallel` performance. This is the
**how**; [`BENCH_PROTOCOL.md`](BENCH_PROTOCOL.md) is the **why** (why one process,
why interleaved, why min-over-reps) and defines the pass/fail bars. Read this
first, reach for the protocol when you want the rationale.

## What the harness does, in one paragraph

Three implementations of each BLAS routine — `par` (our OpenMP overlay), `ob` (an
OpenBLAS-derived reference), and `mig` (the gfortran netlib migration) — are
compiled into **one** binary with their symbols namespaced (`par_`/`ob_`/`mig_`),
then timed **interleaved per rep on the same buffers**. Comparing legs that run
microseconds apart in the same process at the same CPU frequency on the same
pages makes the `par/ref` ratio robust to the systematic error (turbo/DVFS state,
page placement) that poisoned the old cross-process harness. We report **bare
wall time (ns/call)** and the ratio **par / reference, smaller = faster**.

## Prerequisites (once)

1. **An `eplinalg` reference on `CMAKE_PREFIX_PATH`** for each precision you'll
   time — see the top-level [`README.md`](../../README.md) "Running the tests /
   bench" for the per-target install. `e`→kind10, `q`→kind16, `m`→multifloats.
2. **Configure with tests enabled** (the default). The harness links the *test*
   archives, so `-DBUILD_TESTING=ON` must be in effect:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_PREFIX_PATH=/opt/eplinalg-e   # ;/opt/eplinalg-q;/opt/eplinalg-m as needed
   ```
   The configure log must say `tests enabled for target <t>` for every family you
   want to sweep — otherwise the `mig` archive and refblas below won't build.
3. **Toolchain on `PATH`**: `gcc`/`g++`, `gfortran` (the `mig` leg and `-lgfortran`),
   `libquadmath` (for `q` and `m`), `fypp` (driver templates), `taskset`
   (util-linux, for core pinning), `objcopy`/`nm`/`ar` (binutils, for namespacing).
   For the **`m` family only**, `gfortran-15`/`g++-15` are the default compilers
   used to rebuild the double-double runtime non-LTO (override with
   `MF_FC`/`MF_CXX`) — see [Troubleshooting](#troubleshooting).

You don't build the archives by hand — `run_dual.sh` does it (step 1 of its
pipeline) to defeat the [stale-archive trap](#troubleshooting).

## The three things you'll actually run

### 1. Re-time one routine and refresh the committed scoreboard — the everyday cycle

```bash
benchmark/dual/update_routine.sh e etbsv          # one routine
benchmark/dual/update_routine.sh q qtrmm,qtrmv    # a few
```

This rebuilds the archives, re-times just those routines into the full-surface
`results_<fam>` dir, and re-renders `doc/dev/benchmark/results.md` from **all**
three families' data — every other cell is carried over from the last sweep
untouched (the results are one file per routine and the renderer is a pure
reduction). This is what you use after changing a kernel: fix `etbsv`, re-time
`etbsv`, commit the updated `.md`.

### 2. Sweep a whole family — the console board

```bash
benchmark/dual/run_dual.sh e                       # all 75 kind10 routines
benchmark/dual/run_dual.sh q qsyrk,qsyr2k          # just these two
```

End-to-end for one family: build → namespace → generate drivers → compile → run
OMP=1 (1 core) and OMP=4 (4 cores) at `REPS` → print the console scoreboard
(`agg_dual.py`). Raw data lands in `workspace/files/gap5/nsbench/results/`.
Use this for ad-hoc exploration; use `update_routine.sh` when you want the
committed `.md` updated.

### 3. Re-render the committed markdown from existing data

```bash
benchmark/dual/render_scoreboard.py                # -> doc/dev/benchmark/results.md
```

Pure reduction over `results_{m,e,q}/` — no timing. Run it if you edited the
renderer or want to regenerate the doc without re-timing anything.

> **CMake-integrated path.** `cmake --workflow --preset sweep` runs `run_dual.sh`
> once per family through ctest (`dual_sweep_{e,q,m}`, label `sweep`). Same
> harness, wrapped for CI-style invocation.

## Environment knobs

All three scripts honor these (defaults in parens):

| Var | Default | Meaning |
|---|---|---|
| `REPS` | `40` | reps per cell. **Keep ≥ 40** for sub-2% verdicts; min-of-10 fabricates 1–2% phantom gaps. |
| `CORE1` | `2` | core for the OMP=1 (serial) run. |
| `CORE4` | `2-5` | cores for the OMP=4 run. |
| `SKIP_BUILD` | unset | reuse existing namespaced archives — **only** when you know no par/ob/mig source changed (else you measure stale code). |
| `NOAGG` | unset | skip the console board (`update_routine.sh` sets this and renders the `.md` instead). |
| `OUT` / `NSDIR` | scratch tree | where raw data / namespaced archives land. |
| `MF_FC` / `MF_CXX` | `gfortran-15` / `g++-15` | compilers for the `m`-family non-LTO runtime rebuild. |
| `BAR` | `1.02` render / `1.05` agg | flag threshold (par/ref above this is a ⚠). |

Example — a fast, cheap probe of one routine:
```bash
REPS=20 benchmark/dual/run_dual.sh e enrm2
```

## Reading the scoreboard

Every cell is `(routine, key, N)`; the number is **par / reference wall time,
smaller = faster**. The bars (these **override** any generic "par ≤ ob" rule):

- **Serial** — `par1 ≤ min(ob1, mig1)`. par must beat **both** the OpenBLAS clone
  and the netlib migration, whichever is faster. The **`leg` column** tells you
  which one binds; `mig` (netlib triple-loop) is often the faster serial leg on
  stride-1 Transpose kernels, so `par ≤ ob` alone is **not** a serial pass.
- **OMP=4** — `par4 ≤ ob4`.
- **Scaling** — `par4 / par1` (report only; `< 1` = threading helps).

Cells are flagged at **par/ref > 1.02**. The reps≥40 in-process harness is
trustworthy to sub-2%, so **1.00–1.02 is the noise band** — don't chase it. In
the committed `.md`, ✅/⚠ is per routine (⚠ = at least one flagged cell), with a
collapsible appendix listing every flagged cell worst-first.

## Machine hygiene (this is what makes the numbers real)

- **Idle box.** Verify nothing else is running (`uptime`, `pgrep -af taskset`)
  before you time. A background compile skews the ratio.
- **Never run two pinned sweeps at once.** They contend and poison *both* — the
  single most common way to fabricate a gap. One sweep at a time, full stop.
- **Pinning is automatic** (`taskset`, OMP=1→one core, OMP=4→four); a warmup spin
  ramps turbo to steady state before the first timed call.
- **Wall clock only** (`CLOCK_MONOTONIC`); the harness never uses `cycles:u`
  across thread counts (frequency-variant).

## Time expectations

- One routine, both OMP settings, reps=40: **tens of seconds** — but big-`N`
  kind16 (`q`, software `__float128`) cells run much longer.
- A full family (`run_dual.sh q`): **hours**.
- The full three-family reps=40 surface: **~days** — the `q` `mig` leg on big-`N`
  L3/L1 is the long pole. The box is pinned exclusively the whole time.
  Consider `REPS=20` for a faster full refresh (trustworthy to ~2–3% only).

## Where things land

- **Raw data + generated drivers** → `workspace/files/gap5/nsbench/` (gitignored).
  `results_<fam>/` is the committed full surface `update_routine.sh` writes into;
  `results/` is `run_dual.sh`'s ad-hoc dir.
- **Committed** → only `doc/dev/benchmark/results.md`. Never commit anything under
  `benchmark/` beyond the scripts, and never the raw `.txt`.

## Troubleshooting

- **Numbers identical before and after a code change** → stale archive. The
  consumer binary didn't relink. `run_dual.sh` rebuilds by default; make sure you
  did **not** pass `SKIP_BUILD=1`.
- **`refblas (libblas-gfortran) not found` / `mig archive missing`** → you
  configured without `-DBUILD_TESTING=ON`, or the family's eplinalg reference
  isn't on `CMAKE_PREFIX_PATH`. Re-check the configure log for `tests enabled for
  target <t>`.
- **`m` family link errors about `lto1` / bytecode version** → the netlib `mig`
  leg uses the gfortran-15-pinned multifloats runtime, which can't link into a
  newer-gcc driver. `run_dual.sh` recompiles that runtime **non-LTO** with
  `gfortran-15`/`g++-15`; if those aren't your default `-15`, set `MF_FC`/`MF_CXX`.
- **A "gap" you can't reproduce** → almost always low reps (`< 40`) or contention
  (something else ran during the sweep). Re-run at `REPS=40` on an idle box before
  believing any sub-2% verdict. Unbounded fills that overflow to NaN/denormal also
  masquerade as gaps — the drivers force diagonal dominance to avoid this.

## Files

| File | Role |
|---|---|
| `run_dual.sh` | end-to-end sweep for one family (build → ns → gen → run → agg). |
| `update_routine.sh` | re-time one/few routines + re-render the committed `.md`. |
| `render_scoreboard.py` | pure reduction of `results_{m,e,q}/` → `doc/dev/benchmark/results.md`. |
| `agg_dual.py` | console scoreboard + the shared `build_rows` ratio math. |
| `nsbuild.sh` | namespace the par/ob/mig archives into `lib_{par,ob,mig}_ns.a`. |
| `../../benchmark/gen_dual_harnesses.py` | emit the per-routine dual drivers (`--family`, `--routines`, `--list`). |
| `BENCH_PROTOCOL.md` | the rationale + bars + reps discipline. |
