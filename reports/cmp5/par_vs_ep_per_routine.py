#!/usr/bin/env python3
"""For each routine, emit max(par/ep) at OMP=1 and OMP=4.

Used to build the `par>ep` column in src/epopenblas/CHECKLIST.md.
"""
import csv
from collections import defaultdict
from pathlib import Path

from columns import SUBJECTS

CMP = Path(__file__).parent / "cmp5.tsv"
THRESH = 1.10
EP1, EP4, P1, P4 = SUBJECTS


def pf(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


by_rt_1 = defaultdict(float)
by_rt_4 = defaultdict(float)
by_rt_1_n = defaultdict(int)
by_rt_4_n = defaultdict(int)

for r in csv.DictReader(CMP.open(), delimiter="\t"):
    rt = r["routine"]
    ep1 = pf(r[EP1.tsv_col]);    p1 = pf(r[P1.tsv_col])
    ep4 = pf(r[EP4.tsv_col]);    p4 = pf(r[P4.tsv_col])
    if ep1 and p1:
        ratio = p1 / ep1
        if ratio >= THRESH:
            by_rt_1_n[rt] += 1
        by_rt_1[rt] = max(by_rt_1[rt], ratio)
    if ep4 and p4:
        ratio = p4 / ep4
        if ratio >= THRESH:
            by_rt_4_n[rt] += 1
        by_rt_4[rt] = max(by_rt_4[rt], ratio)

routines = sorted(set(by_rt_1) | set(by_rt_4))
print(f"{'routine':<10} {'omp1 max':>10} {'omp1 wins':>10} {'omp4 max':>10} {'omp4 wins':>10}  cell")
for rt in routines:
    m1 = by_rt_1.get(rt, 0.0)
    m4 = by_rt_4.get(rt, 0.0)
    n1 = by_rt_1_n.get(rt, 0)
    n4 = by_rt_4_n.get(rt, 0)
    c1 = f"{m1:.2f}×" if m1 >= THRESH else "—"
    c4 = f"{m4:.2f}×" if m4 >= THRESH else "—"
    cell = f"{c1} / {c4}" if (m1 >= THRESH or m4 >= THRESH) else "—"
    print(f"{rt:<10} {m1:>9.2f}× {n1:>10d} {m4:>9.2f}× {n4:>10d}  {cell}")
