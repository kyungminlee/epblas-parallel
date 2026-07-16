# Archived cross-process cmp5 reports — HISTORICAL, not current

These Markdown files are **frozen historical bench reports** produced by the
retired cross-process `cmp5` harness (separate par/ob/mig binaries, timed in
different processes). They are kept for the record only — the numbers reflect
the state of the tree when each was written and are **not** regenerated.

**Do not cite these for current verdicts.** The cross-process method carried a
layout/frequency bias that the in-process namespaced dual-link harness was
built to remove. The current benchmarking method, protocol, and scoreboard
live under `benchmark/dual/`:

- `benchmark/dual/run_dual.sh`   — end-to-end sweep (build → namespace → generate → run → aggregate)
- `benchmark/dual/agg_dual.py`   — scoreboard (par1 ≤ min(ob1,mig1); par4 ≤ ob4; par4/par1 scaling)
- `benchmark/dual/BENCH_PROTOCOL.md` — the current protocol

`BENCH_PROTOCOL.md` in this directory is the **old** cross-process protocol,
superseded by the one above.
