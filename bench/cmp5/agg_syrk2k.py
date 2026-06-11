#!/usr/bin/env python3
"""Aggregate the focused qsyrk/qsyr2k 5-way sweep into a per-routine/size table.

min-over-reps for each of the 4 subject columns (ep omp1/omp4, par omp1/omp4)
plus the shared migrated reference. Reports bare wall time (ns/call) and the key
ratios; flags regressions (par slower than the ep openblas reference, or par
slower than migrated serial, by >5%).
"""
import csv
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).parent / ".." / ".." / "workspace" / "files" / "gap5" / "cmp5_syrk2k_raw.tsv"
RUNS = ["epopenblas-omp1", "epopenblas-omp4", "parallel-blas-omp1", "parallel-blas-omp4"]

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

REG = 1.05
worst = {}
print(f"{'routine':<8} {'size':>6} | {'ep1':>11} {'ep4':>11} {'par1':>11} {'par4':>11} {'mig':>11} | "
      f"{'p1/ep1':>7} {'p4/ep4':>7} {'p1/mig':>7} {'p4/mig':>7} {'p4/p1':>6}  flags")
for rt in routines:
    for (r2, sz) in [k for k in keys if k[0] == rt]:
        g = lambda rid: subj.get((rt, sz, rid), float("nan"))
        ep1, ep4, p1, p4 = g(RUNS[0]), g(RUNS[1]), g(RUNS[2]), g(RUNS[3])
        m = mig.get((rt, sz), float("nan"))
        def rat(a, b):
            return (a / b) if (b and b == b and a == a and b > 0) else float("nan")
        p1ep1, p4ep4 = rat(p1, ep1), rat(p4, ep4)
        p1m, p4m, p4p1 = rat(p1, m), rat(p4, m), rat(p4, p1)
        flags = []
        if p1ep1 == p1ep1 and p1ep1 > REG: flags.append(f"par1>ep1 {p1ep1:.2f}")
        if p4ep4 == p4ep4 and p4ep4 > REG: flags.append(f"par4>ep4 {p4ep4:.2f}")
        if p1m == p1m and p1m > REG: flags.append(f"par1>mig {p1m:.2f}")
        fl = "; ".join(flags) if flags else "ok"
        if flags:
            worst[rt] = max(worst.get(rt, 0), max((p1ep1 if p1ep1==p1ep1 else 0),
                                                  (p4ep4 if p4ep4==p4ep4 else 0),
                                                  (p1m if p1m==p1m else 0)))
        print(f"{rt:<8} {sz:>6} | {ep1:>11.1f} {ep4:>11.1f} {p1:>11.1f} {p4:>11.1f} {m:>11.1f} | "
              f"{p1ep1:>7.3f} {p4ep4:>7.3f} {p1m:>7.3f} {p4m:>7.3f} {p4p1:>6.3f}  {fl}")
    print()

print("=== regression summary (flagged routines, worst ratio) ===")
if worst:
    for rt, w in sorted(worst.items(), key=lambda x: -x[1]):
        print(f"  {rt}: worst flagged ratio {w:.3f}")
else:
    print("  none — no par-slower-than-baseline cell beyond 5% at any size/omp")
