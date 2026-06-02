#!/usr/bin/env python3
"""List (routine, key, size) where parallel-blas outperforms epopenblas.

Compares same-OMP pairs:
  par-omp1 vs ep-omp1
  par-omp4 vs ep-omp4

All cmp5.tsv values are bare wall time (ns/call, smaller = faster). The
reported ratio is par_ns / ep_ns; a win is par ≥ 10% faster, i.e.
par/ep ≤ 1/1.10 ≈ 0.909. Ratios in the 0.909–1.0 band are noise at the
short timescales these benches run at.
"""
import csv
from collections import defaultdict
from pathlib import Path

from columns import SUBJECTS

HERE = Path(__file__).parent
CMP = HERE / "cmp5.tsv"
OUT = HERE / "par_wins.md"
WIN = 1.0 / 1.10   # par/ep wall ratio at/below this ⇒ par ≥ 10% faster
EP1, EP4, P1, P4 = SUBJECTS


def pf(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(CMP.open(), delimiter="\t"))

    omp1 = []  # (ratio, routine, key, size, ep, par)  ratio = par_ns/ep_ns
    omp4 = []
    for r in rows:
        ep1 = pf(r[EP1.tsv_col]);    p1 = pf(r[P1.tsv_col])
        ep4 = pf(r[EP4.tsv_col]);    p4 = pf(r[P4.tsv_col])
        if ep1 and p1 and p1 / ep1 <= WIN:
            omp1.append((p1/ep1, r["routine"], r["key"], int(r["size"]), ep1, p1))
        if ep4 and p4 and p4 / ep4 <= WIN:
            omp4.append((p4/ep4, r["routine"], r["key"], int(r["size"]), ep4, p4))

    # Smallest ratio = biggest par win.
    omp1.sort()
    omp4.sort()

    out = []
    out.append("# Where parallel-blas beats epopenblas")
    out.append("")
    out.append(f"Threshold: par/ep ≤ {WIN:.3f}× wall time (par ≥ 10% faster; anything larger is sub-noise on these benches). Values are ns/call, smaller = faster.")
    out.append(f"Same-OMP comparison only. Source: `cmp5.tsv` ({len(rows)} rows).")
    out.append("")

    # --- per-routine summary ---
    by_rt_1 = defaultdict(list)
    by_rt_4 = defaultdict(list)
    for ratio, rt, k, n, ep, p in omp1: by_rt_1[rt].append(ratio)
    for ratio, rt, k, n, ep, p in omp4: by_rt_4[rt].append(ratio)

    out.append("## Per-routine win counts")
    out.append("")
    out.append("Number of (key, size) rows where par-blas beat epopenblas by ≥10%, and the median/best par/ep wall ratio over those wins (smaller = bigger win).")
    out.append("")
    out.append("| routine | omp1 wins | omp1 med | omp1 best | omp4 wins | omp4 med | omp4 best |")
    out.append("|---------|----------:|---------:|----------:|----------:|---------:|----------:|")
    all_rt = sorted(set(by_rt_1) | set(by_rt_4))
    for rt in all_rt:
        r1 = by_rt_1.get(rt, [])
        r4 = by_rt_4.get(rt, [])
        if not r1 and not r4:
            continue
        def fmt(rs):
            if not rs:
                return ("0", "—", "—")
            from statistics import median
            return (str(len(rs)), f"{median(rs):.2f}×", f"{min(rs):.2f}×")
        a1 = fmt(r1); a4 = fmt(r4)
        out.append(f"| {rt} | {a1[0]} | {a1[1]} | {a1[2]} | {a4[0]} | {a4[1]} | {a4[2]} |")
    out.append("")

    # --- full row-level lists ---
    def emit(lbl, recs):
        out.append(f"## {lbl} (par/ep wall ratio, smaller = bigger win) — {len(recs)} rows")
        out.append("")
        if not recs:
            out.append("_(none)_")
            out.append("")
            return
        out.append("| ratio | routine | key | N | ep ns | par ns |")
        out.append("|------:|---------|-----|--:|------:|-------:|")
        for ratio, rt, k, n, ep, p in recs:
            out.append(f"| {ratio:.2f}× | {rt} | {k} | {n} | {ep:.1f} | {p:.1f} |")
        out.append("")

    emit("OMP=1 wins", omp1)
    emit("OMP=4 wins", omp4)

    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}  (omp1 wins: {len(omp1)}, omp4 wins: {len(omp4)})")


if __name__ == "__main__":
    main()
