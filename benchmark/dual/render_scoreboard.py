#!/usr/bin/env python3
"""Render the dual-link perf scoreboard as a committed markdown doc.

Reads the same per-routine results dirs that agg_dual.py consumes (reusing its
build_rows so the ratio math lives in ONE place) and writes a durable markdown
snapshot: pass rates + a per-routine summary (one row per routine, worst-first)
+ a flagged-cell appendix listing every cell where par loses.

Because the underlying results are one file per routine and build_rows is a pure
reduction, this regenerates correctly after re-timing a SINGLE routine — that
routine's rows update and every other row is carried over from the last sweep.
The one-routine cycle is benchmark/dual/update_routine.sh.

Usage:
    render_scoreboard.py [out.md] [results_dir ...]
        out.md        default: doc/dev/benchmark/results.md
        results_dir   default: workspace/files/gap5/nsbench/results_{m,e,q}
                      (each dir's family is inferred from its routines' first letter)
Env: BAR (flag threshold, default 1.02 — the reps>=40 sub-2% trust floor).
"""
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agg_dual import build_rows  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BAR = float(os.environ.get("BAR", "1.02"))

FAM_NAME = {"m": "multifloats (double-double)", "e": "kind10 (fp80)",
            "q": "kind16 (__float128)"}


def fam_of(rows):
    """Infer family letter from the routines present (first non-i letter → m/e/q)."""
    for r in rows:
        c = r["rt"][0]
        c = r["rt"][1] if c == "i" and len(r["rt"]) > 1 else c
        if c in "mw":
            return "m"
        if c in "ey":
            return "e"
        if c in "qx":
            return "q"
    return "?"


def pass_rate(vals, bar):
    vals = [v for v in vals if v == v]
    return (sum(1 for v in vals if v <= bar) / len(vals) * 100) if vals else float("nan")


def per_routine(rows):
    """routine -> dict(ncells, ser_worst, ser_leg, ser_cell, omp_worst, omp_cell, nfail)."""
    agg = {}
    for r in rows:
        a = agg.setdefault(r["rt"], dict(ncells=0, ser_worst=0.0, ser_leg="", ser_cell="",
                                         omp_worst=0.0, omp_cell="", nfail=0))
        a["ncells"] += 1
        cell = f"{r['key']}/{r['sz']}"
        if r["p1bar"] == r["p1bar"] and r["p1bar"] > a["ser_worst"]:
            a["ser_worst"], a["ser_leg"], a["ser_cell"] = r["p1bar"], r["sref"], cell
        if r["p4bar"] == r["p4bar"] and r["p4bar"] > a["omp_worst"]:
            a["omp_worst"], a["omp_cell"] = r["p4bar"], cell
        if (r["p1bar"] == r["p1bar"] and r["p1bar"] > BAR) or \
           (r["p4bar"] == r["p4bar"] and r["p4bar"] > BAR):
            a["nfail"] += 1
    return agg


def render_family(out, fam, rows):
    agg = per_routine(rows)
    sp = pass_rate([r["p1bar"] for r in rows], BAR)
    op = pass_rate([r["p4bar"] for r in rows], BAR)
    ncells = len(rows)
    nflag = sum(1 for a in agg.values() if a["nfail"])
    out.append(f"## {fam} — {FAM_NAME.get(fam, fam)}\n")
    out.append(f"{ncells} cells, {len(agg)} routines.  "
               f"**Pass@{BAR:.2f}: serial {sp:.1f}% · omp4 {op:.1f}%.**  "
               f"{nflag} routine(s) with ≥1 flagged cell.\n")
    out.append("| routine | cells | serial worst (par/min, leg) | omp4 worst (par/ob4) | status |")
    out.append("|---|--:|---|---|:--:|")
    # worst-first: routines with fails on top, sorted by max(serial,omp4) worst ratio
    def sortkey(item):
        a = item[1]
        return -(max(a["ser_worst"], a["omp_worst"]))
    for rt, a in sorted(agg.items(), key=sortkey):
        bad = a["nfail"] > 0
        ser = (f"{a['ser_worst']:.3f} {a['ser_leg']} @{a['ser_cell']}"
               if a["ser_worst"] > BAR else f"{a['ser_worst']:.3f}")
        omp = (f"{a['omp_worst']:.3f} @{a['omp_cell']}"
               if a["omp_worst"] > BAR else f"{a['omp_worst']:.3f}")
        out.append(f"| {'**'+rt+'**' if bad else rt} | {a['ncells']} | {ser} | {omp} | "
                   f"{'⚠' if bad else '✅'} |")
    out.append("")
    # flagged-cell appendix
    flagged = [r for r in rows if (r["p1bar"] == r["p1bar"] and r["p1bar"] > BAR)
               or (r["p4bar"] == r["p4bar"] and r["p4bar"] > BAR)]
    if flagged:
        out.append(f"<details><summary>{fam}: {len(flagged)} flagged cells "
                   f"(par/ref > {BAR:.2f}, smaller=faster)</summary>\n")
        out.append("| routine | key | N | par1 | ob1 | mig1 | par4 | ob4 | p1/min | p4/ob4 | leg |")
        out.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")

        def fz(x):
            return f"{x:,.0f}" if x == x else "-"

        def fr(x):
            return f"{x:.3f}" if x == x else "-"
        for r in sorted(flagged, key=lambda r: -max(
                r["p1bar"] if r["p1bar"] == r["p1bar"] else 0,
                r["p4bar"] if r["p4bar"] == r["p4bar"] else 0)):
            out.append(f"| {r['rt']} | {r['key']} | {r['sz']} | {fz(r['par1'])} | {fz(r['ob1'])} "
                       f"| {fz(r['mig1'])} | {fz(r['par4'])} | {fz(r['ob4'])} | {fr(r['p1bar'])} "
                       f"| {fr(r['p4bar'])} | {r['sref']} |")
        out.append("\n</details>\n")


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "doc/dev/benchmark/results.md"
    if len(sys.argv) > 2:
        dirs = [Path(d) for d in sys.argv[2:]]
    else:
        ns = ROOT / "workspace/files/gap5/nsbench"
        dirs = [ns / f"results_{f}" for f in ("m", "e", "q")]

    fam_rows = {}
    for d in dirs:
        rows = build_rows(d)
        if rows:
            fam_rows[fam_of(rows)] = rows

    if not fam_rows:
        sys.exit("no results found in: " + ", ".join(str(d) for d in dirs))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = []
    out.append("# Dual-link perf scoreboard\n")
    out.append(f"_Generated {stamp} by `benchmark/dual/render_scoreboard.py`._\n")
    out.append("All values are **bare wall time (ns/call)**, ratio = **par / reference, "
               "smaller = faster**. Bars (OVERRIDE defaults): serial `par1 ≤ min(ob1, mig1)`; "
               f"omp4 `par4 ≤ ob4`. Cells are flagged at **par/ref > {BAR:.2f}** (the reps≥40 "
               "in-process harness is trustworthy to sub-2%; 1.00–1.02 is the noise band). "
               "`leg` = which serial reference binds (`mig` = netlib triple-loop, `ob1` = "
               "OpenBLAS clone). See `benchmark/dual/BENCH_PROTOCOL.md`.\n")
    # headline table
    out.append("| family | cells | serial pass@%s | omp4 pass@%s |" % (f"{BAR:.2f}", f"{BAR:.2f}"))
    out.append("|---|--:|--:|--:|")
    for fam in ("m", "e", "q"):
        if fam in fam_rows:
            rows = fam_rows[fam]
            out.append(f"| {fam} | {len(rows)} | {pass_rate([r['p1bar'] for r in rows], BAR):.1f}% "
                       f"| {pass_rate([r['p4bar'] for r in rows], BAR):.1f}% |")
    out.append("")
    for fam in ("m", "e", "q"):
        if fam in fam_rows:
            render_family(out, fam, fam_rows[fam])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"wrote {out_path}  ({len(fam_rows)} families)")


if __name__ == "__main__":
    main()
