#!/usr/bin/env python3
"""Aggregate the 3-way multifloats parallelization-gap sweep.

min-over-reps for par-omp1 / par-omp4 plus the shared migrated reference.
Bare wall time (ns/call); ratios SMALLER = faster. No OpenBLAS leg for DD.

  p1/mig = par-serial / migrated-serial   (serial speedup; <1 = par faster)
  p4/mig = par-omp4   / migrated-serial   (end-to-end win)
  p4/p1  = par-omp4   / par-serial        (self-scaling; ~0.25 = 4x)

Flags a serial REGRESSION only if p1/mig worsens vs an optional --before file.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "cmp5_gap_raw.tsv"
RUNS = ["par-omp1", "par-omp4"]

subj = defaultdict(lambda: float("inf"))
mig = defaultdict(lambda: float("inf"))
for r in csv.DictReader(RAW.open(), delimiter="\t"):
    rt, sz, rid = r["routine"], int(r["size"]), r["run_id"]
    try:
        s = float(r["subject_ns"]); m = float(r["migrated_ns"])
    except ValueError:
        continue
    if s > 0:
        subj[(rt, sz, rid)] = min(subj[(rt, sz, rid)], s)
    if m > 0:
        mig[(rt, sz)] = min(mig[(rt, sz)], m)

keys = sorted({(rt, sz) for (rt, sz, _) in subj})
routines = sorted({rt for rt, _ in keys})

print(f"{'routine':<8} {'size':>7} | {'par1':>12} {'par4':>12} {'mig':>12} | "
      f"{'p1/mig':>7} {'p4/mig':>7} {'p4/p1':>6}")
for rt in routines:
    for (r2, sz) in [k for k in keys if k[0] == rt]:
        g = lambda rid: subj.get((rt, sz, rid), float("nan"))
        p1, p4 = g(RUNS[0]), g(RUNS[1])
        m = mig.get((rt, sz), float("nan"))
        rat = lambda a, b: (a / b) if (b and b == b and a == a and b > 0) else float("nan")
        print(f"{rt:<8} {sz:>7} | {p1:>12.1f} {p4:>12.1f} {m:>12.1f} | "
              f"{rat(p1,m):>7.3f} {rat(p4,m):>7.3f} {rat(p4,p1):>6.3f}")
    print()
