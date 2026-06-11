#!/usr/bin/env python3
"""Per-(routine,key,size) before/after comparator for the tpsv threading work.

min-over-reps. Bare wall ns/call, ratios SMALLER = faster.
  p1/mig = par-serial / migrated   (serial parity — must NOT regress vs before)
  p4/p1  = par-omp4   / par-serial  (self-scaling; <1 means threading helps)

Usage: agg_tpsv.py <after_raw.tsv> [before_raw.tsv]
Only plain keys (incx==1, no '/x' suffix) can thread; '/x2','/x-1' fall to serial.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

def load(path):
    subj = defaultdict(lambda: float("inf"))   # (rt,key,sz,rid) -> ns
    mig  = defaultdict(lambda: float("inf"))    # (rt,key,sz) -> ns
    for r in csv.DictReader(Path(path).open(), delimiter="\t"):
        rt, key, sz, rid = r["routine"], r["key"], int(r["size"]), r["run_id"]
        try:
            s = float(r["subject_ns"]); m = float(r["migrated_ns"])
        except ValueError:
            continue
        if s > 0: subj[(rt, key, sz, rid)] = min(subj[(rt, key, sz, rid)], s)
        if m > 0: mig[(rt, key, sz)] = min(mig[(rt, key, sz)], m)
    return subj, mig

after_path = sys.argv[1]
before_path = sys.argv[2] if len(sys.argv) > 2 else None
asubj, amig = load(after_path)
bsubj, bmig = (load(before_path) if before_path else (None, None))

keys = sorted({(rt, k, sz) for (rt, k, sz, _) in asubj})
rat = lambda a, b: (a / b) if (b and b == b and a == a and b > 0) else float("nan")

hdr = f"{'routine':<7} {'key':<9} {'N':>5} | {'par1':>10} {'par4':>10} {'mig':>10} | {'p1/mig':>7} {'p4/p1':>6}"
if before_path: hdr += f" | {'Δp1/mig':>8} {'b:p4/p1':>7}"
print(hdr)
cur = None
for (rt, k, sz) in keys:
    if rt != cur: print(); cur = rt
    p1 = asubj.get((rt, k, sz, "par-omp1"), float("nan"))
    p4 = asubj.get((rt, k, sz, "par-omp4"), float("nan"))
    m  = amig.get((rt, k, sz), float("nan"))
    line = f"{rt:<7} {k:<9} {sz:>5} | {p1:>10.1f} {p4:>10.1f} {m:>10.1f} | {rat(p1,m):>7.3f} {rat(p4,p1):>6.3f}"
    if before_path:
        bp1 = bsubj.get((rt, k, sz, "par-omp1"), float("nan"))
        bp4 = bsubj.get((rt, k, sz, "par-omp4"), float("nan"))
        bm  = bmig.get((rt, k, sz), float("nan"))
        d = rat(p1, m) - rat(bp1, bm)
        line += f" | {d:>+8.3f} {rat(bp4,bp1):>7.3f}"
    print(line)
