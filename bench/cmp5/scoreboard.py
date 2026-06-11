#!/usr/bin/env python3
"""Build ONE ranked current scoreboard CSV from a set of gap5 raw sweeps.

Consolidates the latest post-fix raw files (newest file wins per routine),
computes the per-rep paired stats from agg_gap5, ranks every cell by the worse
of its two median ratios (par1/ob1, par4/ob4), and writes a CSV:

    routine,key,size,par1_ns,par4_ns,ob1_ns,ob4_ns,
    med_p1_o1,ci_lo_p1,ci_hi_p1,med_p4_o4,ci_lo_p4,ci_hi_p4,
    min_p1_o1,min_p4_o4,worst,slow,flags,src

Usage:
    scoreboard.py OUT.csv RAW1.tsv [RAW2.tsv ...]
    scoreboard.py            # defaults: workspace/files/gap5/scoreboard.csv
                             #  over the files listed in DEFAULT_FILES below.
`slow` is 1 when a per-rep ratio CI lower bound > 1.0 (genuinely slower than ob).
"""
import sys, os, csv
import agg_gap5 as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "workspace", "files", "gap5"))

# Current authoritative post-fix raw files, oldest→newest. A routine present in
# a later file supersedes its earlier rows (task-12 fixes beat the task-11
# backstop). Missing files are skipped with a warning.
DEFAULT_FILES = [
    "cmp5_task11_backstop_raw.tsv",
    "egemm_gap5_after_raw.tsv",
    "gap5_argmax_raw.tsv",
    "mgemmtr_gap5_after_raw.tsv",
    "wgemmtr_gap5_after_raw.tsv",
    "mtrsm_gap5_after_raw.tsv",
    "wtbmv_gap5_after_raw.tsv",
    "whemv_after_raw.tsv",
]


def resolve(files):
    out = []
    for f in files:
        for cand in (f, os.path.join(HERE, f), os.path.join(OUT_DIR, f)):
            if os.path.exists(cand):
                out.append(cand); break
        else:
            print(f"# WARN missing {f}", file=sys.stderr)
    return out


def main():
    args = sys.argv[1:]
    if args and args[0].endswith(".csv"):
        out_csv = args[0]; raws = args[1:] or DEFAULT_FILES
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_csv = os.path.join(OUT_DIR, "scoreboard.csv")
        raws = args or DEFAULT_FILES
    paths = resolve(raws)
    if not paths:
        sys.exit("no raw files found")

    # owner map for the `src` column (basename of the file each routine came from)
    owner = {}
    for p in paths:
        with open(p) as f:
            next(f)
            for line in f:
                t = line.split("\t", 3)
                if len(t) >= 3:
                    owner[t[2]] = os.path.basename(p)

    series, _ = A.load(paths)
    rows = []
    for k in series:
        st = A.cell_stats(series[k])
        flags, slow, rm1, rm4 = A.cell_flags(st)
        worst = A.worst_key((k, st))
        routine, key, size = k
        c1lo, c1hi = st["r1_ci"]; c4lo, c4hi = st["r4_ci"]
        rows.append({
            "routine": routine, "key": key, "size": size,
            "par1_ns": round(st["p1"], 1), "par4_ns": round(st["p4"], 1),
            "ob1_ns": round(st["o1"], 1), "ob4_ns": round(st["o4"], 1),
            "med_p1_o1": round(st["r1_med"], 4), "ci_lo_p1": round(c1lo, 4), "ci_hi_p1": round(c1hi, 4),
            "med_p4_o4": round(st["r4_med"], 4), "ci_lo_p4": round(c4lo, 4), "ci_hi_p4": round(c4hi, 4),
            "min_p1_o1": round(rm1, 4), "min_p4_o4": round(rm4, 4),
            "worst": round(worst, 4), "slow": int(slow),
            "flags": " ".join(flags), "src": owner.get(routine, ""),
        })
    rows.sort(key=lambda r: r["worst"], reverse=True)
    cols = ["routine", "key", "size", "par1_ns", "par4_ns", "ob1_ns", "ob4_ns",
            "med_p1_o1", "ci_lo_p1", "ci_hi_p1", "med_p4_o4", "ci_lo_p4", "ci_hi_p4",
            "min_p1_o1", "min_p4_o4", "worst", "slow", "flags", "src"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    nslow = sum(r["slow"] for r in rows)
    print(f"wrote {out_csv}: {len(rows)} cells, {len(set(r['routine'] for r in rows))} routines, {nslow} SLOW")
    print(f"\ntop 15 worst (worse of med par1/ob1, par4/ob4; smaller=faster):")
    print(f"{'routine':9}{'key':11}{'N':>5}  {'p1/o1':>6} {'p4/o4':>6}  {'slow':>4}  flags")
    for r in rows[:15]:
        print(f"{r['routine']:9}{r['key']:11}{r['size']:5d}  "
              f"{r['med_p1_o1']:6.3f} {r['med_p4_o4']:6.3f}  {r['slow']:>4}  {r['flags']}")


if __name__ == "__main__":
    main()
