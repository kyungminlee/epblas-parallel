#!/usr/bin/env python3
"""Build a human-readable summary of cmp5.tsv.

Sanity-checks the migrated-serial column by comparing the 4 mig_* readings
(should be near-equal since `_serial` is OpenMP-free), then emits:

  1) Per-routine table — for each routine, the median GF/s across (key, size)
     for each of the 5 columns plus the omp4 speedups.
  2) Migrated-baseline drift table — how much the 4 mig_* readings disagree
     (should be small).
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


HERE = Path(__file__).parent
CMP = HERE / "cmp5.tsv"
OUT = HERE / "cmp5_summary.md"


def pf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = []
    with CMP.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)

    by_routine = defaultdict(list)
    for r in rows:
        by_routine[r["routine"]].append(r)

    out = []
    out.append("# 5-way comparison: epopenblas / parallel-blas / migrated — kind10 (REAL/COMPLEX(KIND=10))")
    out.append("")
    out.append(f"- Source: `reports/cmp5/cmp5.tsv` ({len(rows)} (routine,key,size) rows over {len(by_routine)} routines)")
    out.append("- Five variants: epopenblas (overlay) at OMP=1 and OMP=4, parallel-blas (overlay) at OMP=1 and OMP=4, and migrated (Fortran reference, serial baseline).")
    out.append("- All four overlay binaries link the SAME `tests/blas_parallel/perf/target_kind10/perf_<r>.c` source — only the C-overlay symbol differs.")
    out.append("- Same `BLAS_PERF_{ITERS,WARMUP,INCX,INCY}=200/20/1/1`; per-routine default sizes; pinned via `taskset` (P-cores 0 or 0..3).")
    out.append("- migrated-serial = migrated_GFs from the parallel-blas-omp1 run; mig_* columns are sanity readings of the same migrated `_serial` symbol from each of the four runs (expected to be ~equal since `_serial` contains no OpenMP).")
    out.append("")

    # 1. migrated drift sanity check.
    out.append("## Migrated-baseline drift across runs (sanity check)")
    out.append("")
    out.append("`max/min` ratio of the four `mig_*` readings per (routine,key,size). Closer to 1.00 ⇒ the migrated_ symbol's wall time is invariant w.r.t. OMP env / which binary it lived in.")
    out.append("")
    drifts = []
    for r in rows:
        mig = [pf(r[c]) for c in ("mig_par_omp1", "mig_par_omp4", "mig_ep_omp1", "mig_ep_omp4")]
        mig = [m for m in mig if m and m > 0]
        if len(mig) < 4:
            continue
        drifts.append((max(mig) / min(mig), r["routine"], r["key"], int(r["size"])))
    drifts.sort(reverse=True)
    out.append(f"- median drift ratio:  **{median(d[0] for d in drifts):.3f}×**")
    out.append(f"- p95 drift ratio:     **{sorted(d[0] for d in drifts)[int(0.95*len(drifts))]:.3f}×**")
    out.append(f"- max drift ratio:     **{drifts[0][0]:.3f}×**  at {drifts[0][1]}/{drifts[0][2]}/N={drifts[0][3]}")
    out.append("")
    out.append("Top 10 drifters (rows where the 4 migrated readings diverge most):")
    out.append("")
    out.append("| ratio | routine | key | N |")
    out.append("|------:|---------|-----|--:|")
    for ratio, rt, k, n in drifts[:10]:
        out.append(f"| {ratio:.3f} | {rt} | {k} | {n} |")
    out.append("")

    # 2. Per-routine median GF/s across all (key, size).
    out.append("## Per-routine medians (GF/s) — across all (key, size)")
    out.append("")
    out.append("Columns: routine, then median GF/s for each of the 5 variants (`ep` = epopenblas, `par` = parallel-blas, `mig` = migrated Fortran reference), then OMP=4/OMP=1 speedup for each C overlay.")
    out.append("")
    out.append("| routine | ep-omp1 | ep-omp4 | par-omp1 | par-omp4 | mig-serial | ep4/ep1 | par4/par1 |")
    out.append("|---------|--------:|--------:|---------:|---------:|-----------:|--------:|----------:|")
    for routine in sorted(by_routine):
        rr = by_routine[routine]
        ep1 = [pf(r["epopenblas-omp1"])    for r in rr]; ep1 = [v for v in ep1 if v]
        ep4 = [pf(r["epopenblas-omp4"])    for r in rr]; ep4 = [v for v in ep4 if v]
        p1  = [pf(r["parallel-blas-omp1"]) for r in rr]; p1  = [v for v in p1  if v]
        p4  = [pf(r["parallel-blas-omp4"]) for r in rr]; p4  = [v for v in p4  if v]
        mig = [pf(r["migrated-serial"])    for r in rr]; mig = [v for v in mig if v]
        if not (ep1 and ep4 and p1 and p4 and mig):
            continue
        m_ep1 = median(ep1); m_ep4 = median(ep4)
        m_p1 = median(p1);   m_p4 = median(p4);  m_mig = median(mig)
        out.append(f"| {routine} | {m_ep1:.3f} | {m_ep4:.3f} | {m_p1:.3f} | {m_p4:.3f} | {m_mig:.3f} | {m_ep4/m_ep1:.2f}× | {m_p4/m_p1:.2f}× |")

    out.append("")

    # 3. Big-N comparison.
    big_n_routines = defaultdict(list)
    for r in rows:
        n = int(r["size"])
        if n < 512:
            continue
        big_n_routines[r["routine"]].append((n, r))
    out.append("## Per-routine — single (largest) N only")
    out.append("")
    out.append("For each routine, the row at the largest N actually measured. Useful for cases where a serial-OK overlay regresses only at large N (or vice versa).")
    out.append("")
    out.append("| routine | key | N | ep-omp1 | ep-omp4 | par-omp1 | par-omp4 | mig-serial | ep1/mig | par1/mig | par4/mig |")
    out.append("|---------|-----|--:|--------:|--------:|---------:|---------:|-----------:|--------:|---------:|---------:|")
    for routine in sorted(big_n_routines):
        rr = big_n_routines[routine]
        n_max = max(n for n, _ in rr)
        # pick the (routine, key, N=n_max) row; if multiple keys at this N,
        # show the largest-N row with the lexicographically-first key.
        cands = [r for n, r in rr if n == n_max]
        cands.sort(key=lambda r: r["key"])
        r = cands[0]
        ep1 = pf(r["epopenblas-omp1"]); ep4 = pf(r["epopenblas-omp4"])
        p1 = pf(r["parallel-blas-omp1"]); p4 = pf(r["parallel-blas-omp4"])
        mig = pf(r["migrated-serial"])
        if not (ep1 and ep4 and p1 and p4 and mig):
            continue
        out.append(
            f"| {routine} | {r['key']} | {n_max} | {ep1:.3f} | {ep4:.3f} | {p1:.3f} | {p4:.3f} | {mig:.3f} "
            f"| {ep1/mig:.2f}× | {p1/mig:.2f}× | {p4/mig:.2f}× |"
        )
    out.append("")

    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
