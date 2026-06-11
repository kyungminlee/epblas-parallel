#!/usr/bin/env python3
"""5-way before/after comparator for the kind10 gap routines.

min-over-reps. Bare wall ns/call, ratios SMALLER = faster.
Legs (run_id):
  par1 = parallel-blas-omp1 subject_ns   ob1 = epopenblas-omp1 subject_ns
  par4 = parallel-blas-omp4 subject_ns   ob4 = epopenblas-omp4 subject_ns
  mig  = migrated_ns (built into every perf binary)

Reported:
  p1/mig  par serial vs migrated   (serial parity — must NOT regress vs before)
  p4/p1   par self-scaling         (<1 ⇒ threading helps)
  p4/ob4  par-omp4 vs openblas-omp4 (5-way headline; <=1 ⇒ par at least ties ob)

Usage: agg_gapk10.py <after_raw.tsv> [before_raw.tsv]
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

RID = {"parallel-blas-omp1": "par1", "parallel-blas-omp4": "par4",
       "epopenblas-omp1": "ob1", "epopenblas-omp4": "ob4"}

def load(path):
    subj = defaultdict(lambda: float("inf"))   # (rt,key,sz,leg) -> ns
    mig  = defaultdict(lambda: float("inf"))    # (rt,key,sz) -> ns
    for r in csv.DictReader(Path(path).open(), delimiter="\t"):
        leg = RID.get(r["run_id"])
        if leg is None:
            continue
        rt, key, sz = r["routine"], r["key"], int(r["size"])
        try:
            s = float(r["subject_ns"]); m = float(r["migrated_ns"])
        except ValueError:
            continue
        if s > 0: subj[(rt, key, sz, leg)] = min(subj[(rt, key, sz, leg)], s)
        if m > 0: mig[(rt, key, sz)] = min(mig[(rt, key, sz)], m)
    return subj, mig

after_path = sys.argv[1]
before_path = sys.argv[2] if len(sys.argv) > 2 else None
asubj, amig = load(after_path)
bsubj, bmig = (load(before_path) if before_path else (None, None))

keys = sorted({(rt, k, sz) for (rt, k, sz, _) in asubj})
rat = lambda a, b: (a / b) if (b and b == b and a == a and b > 0) else float("nan")

hdr = (f"{'routine':<7} {'key':<9} {'N':>6} | {'par1':>11} {'par4':>11} {'ob4':>11} {'mig':>11} "
       f"| {'p1/mig':>7} {'p4/p1':>6} {'p4/ob4':>7}")
if before_path: hdr += f" | {'Δp1/mig':>8} {'p1/bp1':>7} {'b:p4/p1':>7}"
print(hdr)
cur = None
for (rt, k, sz) in keys:
    if rt != cur: print(); cur = rt
    p1 = asubj.get((rt, k, sz, "par1"), float("nan"))
    p4 = asubj.get((rt, k, sz, "par4"), float("nan"))
    o4 = asubj.get((rt, k, sz, "ob4"), float("nan"))
    m  = amig.get((rt, k, sz), float("nan"))
    line = (f"{rt:<7} {k:<9} {sz:>6} | {p1:>11.1f} {p4:>11.1f} {o4:>11.1f} {m:>11.1f} "
            f"| {rat(p1,m):>7.3f} {rat(p4,p1):>6.3f} {rat(p4,o4):>7.3f}")
    if before_path:
        bp1 = bsubj.get((rt, k, sz, "par1"), float("nan"))
        bp4 = bsubj.get((rt, k, sz, "par4"), float("nan"))
        bm  = bmig.get((rt, k, sz), float("nan"))
        d = rat(p1, m) - rat(bp1, bm)
        line += f" | {d:>+8.3f} {rat(p1,bp1):>7.3f} {rat(bp4,bp1):>7.3f}"
    print(line)
