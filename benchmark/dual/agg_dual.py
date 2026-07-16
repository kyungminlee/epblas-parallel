#!/usr/bin/env python3
"""Scoreboard aggregator for the dual-link (option-2) perf harness.

Consumes the per-routine output of the dual drivers (one file per OMP setting):

    <results>/<routine>.omp1.txt   # par1 ob1 mig1   (OMP_NUM_THREADS=1)
    <results>/<routine>.omp4.txt   # par4 ob4 mig4   (OMP_NUM_THREADS=4)

Each data line is the dual driver row format:

    <routine> <key> <N> <reset_ns> <par_ns> <ob_ns> <mig_ns> <par/ob> <par/mig> <ob/mig>

par_ns/ob_ns/mig_ns are already reset-subtracted, min-over-reps wall time (ns).
Because all three legs are timed interleaved in ONE process on the SAME buffers,
the ratios are free of the cross-process layout/frequency bias the old cmp5
separate-binary harness suffered — so a sub-2% verdict is trustworthy at reps>=40.

Bars (smaller = faster; OVERRIDE the generic defaults):
    Serial   par1 / min(mig1, ob1)   par must beat the faster of netlib & ob clone
    OMP=4    par4 / ob4
    Scaling  par4 / par1             (report only; <1 = threading helps)

Usage:  agg_dual.py [results_dir] [topN] [bar]

Importable surface (used by render_scoreboard.py so the row math lives in ONE
place): load(path), build_rows(results_dir) -> list[dict].
"""
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    """routine.omp{1,4}.txt -> {(routine, key, N): (par_ns, ob_ns, mig_ns)} (min over dup rows)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 7:
            continue
        rt, key, N = f[0], f[1], int(f[2])
        try:
            par, ob, mig = float(f[4]), float(f[5]), float(f[6])
        except ValueError:
            continue
        k = (rt, key, N)
        if k in out:  # keep the faster (defensive; drivers emit one row per cell)
            par = min(par, out[k][0]); ob = min(ob, out[k][1]); mig = min(mig, out[k][2])
        out[k] = (par, ob, mig)
    return out


def build_rows(results_dir):
    """Reduce a results dir's per-routine .txt files to the per-cell row dicts.

    Pure function of the directory contents — re-running it after re-timing a
    single routine reflects that routine and leaves every other cell untouched.
    Returns [] if the dir holds no data (caller decides whether that's fatal).
    """
    results = Path(results_dir)
    omp1, omp4 = {}, {}
    for p in sorted(results.glob("*.omp1.txt")):
        omp1.update(load(p))
    for p in sorted(results.glob("*.omp4.txt")):
        omp4.update(load(p))

    def rat(a, b):
        return (a / b) if (a == a and b == b and b > 0) else float("nan")

    rows = []
    for c in sorted(set(omp1) | set(omp4)):
        par1, ob1, mig1 = omp1.get(c, (float("nan"),) * 3)
        par4, ob4, _ = omp4.get(c, (float("nan"),) * 3)
        serial_refs = [x for x in (ob1, mig1) if x == x and x > 0]
        sref = min(serial_refs) if serial_refs else float("nan")
        sref_name = ("mig" if (mig1 == mig1 and (ob1 != ob1 or mig1 <= ob1)) else "ob1")
        rows.append(dict(
            rt=c[0], key=c[1], sz=c[2], par1=par1, ob1=ob1, mig1=mig1, par4=par4, ob4=ob4,
            p1bar=rat(par1, sref), p4bar=rat(par4, ob4), scale=rat(par4, par1), sref=sref_name))
    return rows


def _cli():
    RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    BAR = float(sys.argv[3]) if len(sys.argv) > 3 else 1.05

    rows = build_rows(RESULTS)
    if not rows:
        sys.exit(f"no *.omp1.txt / *.omp4.txt found under {RESULTS}")

    hdr = (f"{'routine':<9} {'key':>5} {'N':>6} | {'par1':>12} {'ob1':>12} {'mig1':>12} "
           f"{'par4':>12} {'ob4':>12} | {'p1/min':>7} {'p4/ob4':>7} {'p4/p1':>6} {'sref':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def fz(x): return f"{x:>12.0f}" if x == x else f"{'-':>12}"
        def fr(x): return f"{x:>7.3f}" if x == x else f"{'-':>7}"
        print(f"{r['rt']:<9} {r['key']:>5} {r['sz']:>6} | {fz(r['par1'])} {fz(r['ob1'])} {fz(r['mig1'])} "
              f"{fz(r['par4'])} {fz(r['ob4'])} | {fr(r['p1bar'])} {fr(r['p4bar'])} "
              f"{r['scale'] if r['scale']==r['scale'] else float('nan'):>6.3f} {r['sref']:>4}")

    def board(title, field):
        valid = [r for r in rows if r[field] == r[field]]
        if not valid:
            print(f"\n=== {title} — no data ===")
            return
        worst = sorted(valid, key=lambda r: -r[field])[:TOPN]
        nfail = sum(1 for r in valid if r[field] > BAR)
        print(f"\n=== {title} — worst {len(worst)}  (bar {BAR:.2f}; {nfail}/{len(valid)} FAIL) ===")
        for r in worst:
            flag = "  <<< FAIL" if r[field] > BAR else ""
            print(f"  {r['rt']:<9} {r['key']:>5} {r['sz']:>6}   {field}={r[field]:.4f}{flag}")
        return nfail

    nf1 = board("SERIAL  par1/min(mig1,ob1)", "p1bar")
    nf4 = board("OMP=4   par4/ob4", "p4bar")
    print(f"\n=== best (lowest) par4/par1 scaling, top {TOPN} ===")
    for r in sorted([x for x in rows if x["scale"] == x["scale"]], key=lambda r: r["scale"])[:TOPN]:
        print(f"  {r['rt']:<9} {r['key']:>5} {r['sz']:>6}   par4/par1={r['scale']:.4f}  ({1/r['scale']:.2f}x)")

    total_fail = (nf1 or 0) + (nf4 or 0)
    print(f"\nTOTAL FAIL (serial+omp4, bar {BAR:.2f}): {total_fail}")


if __name__ == "__main__":
    _cli()
