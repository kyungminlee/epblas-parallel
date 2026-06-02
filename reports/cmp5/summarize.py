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

from columns import MIG_COLS, MIGRATED_COL, MIGRATED_LABEL, SUBJECTS


HERE = Path(__file__).parent
CMP = HERE / "cmp5.tsv"
OUT = HERE / "cmp5_summary.md"

# Named handles into SUBJECTS for the pair-specific ratio columns (omp4/omp1).
EP1, EP4, P1, P4 = SUBJECTS
MIG_LABEL = MIGRATED_LABEL


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
    out.append(f"# 5-way comparison: {EP1.label.split('-')[0]} / {P1.label.split('-')[0]} / {MIG_LABEL.split('-')[0]} — kind10 (REAL/COMPLEX(KIND=10))")
    out.append("")
    out.append(f"- Source: `reports/cmp5/cmp5.tsv` ({len(rows)} (routine,key,size) rows over {len(by_routine)} routines)")
    out.append(f"- Five variants: `{EP1.label}` and `{EP4.label}` (epblas-openblas overlay), `{P1.label}` and `{P4.label}` (epblas-parallel overlay), and `{MIG_LABEL}` (Fortran reference, serial baseline).")
    out.append("- All four overlay binaries link the SAME `tests/epblas-parallel/perf/target_kind10/perf_<r>.c` source — only the C-overlay symbol differs.")
    out.append("- Same `BLAS_PERF_{ITERS,WARMUP,INCX,INCY}=200/20/1/1`; per-routine default sizes; pinned via `taskset` (P-cores 0 or 0..3).")
    out.append(f"- `{MIG_LABEL}` = migrated_GFs from the `{P1.label}` run; `mig_*` columns are sanity readings of the same migrated `_serial` symbol from each of the four runs (expected to be ~equal since `_serial` contains no OpenMP).")
    out.append("")
    out.append("**Units: all columns are bare wall time in ns/call — smaller is faster.** A `parallel/openblas` wall-time ratio **< 1.0** means parallel is faster; the firm bar is par ≤ ob in every cell. (The separate interleaved per-routine harness reports the same par/ob wall-time ratio, where < 1.0 = par faster — the same direction. See `doc/optimization-findings.md` → \"Reporting convention\".)")
    out.append("")
    out.append("> ⚠️ **Near-parity cells here are a SCREEN, not a verdict — do not read a slightly-above-1.0 ratio as a regression.** This cmp5 is single-shot (one timing per cell, L3 at `iters=10`) and NOT interleaved, so it carries block-drift bias plus, at OMP=4 with small N, thread-pool start-up noise. Any par/ob ratio in roughly the **0.90–1.10 band is inside the noise** and can swing several percent run-to-run. This applies to BOTH the rewritten L3 routines (`esymm` `esyrk` `esyr2k` `etrsm` `etrmm` `ysymm` `yhemm` `yherk` `yher2k` …) AND near-parity L1/L2 routines — e.g. **`yher2` reads ~1.08× here yet is at/below parity under the interleaved harness** (U 0.957–0.977, L ~1.00 wall ratio); likewise treat slightly-above-1.0 single-shot reads for `edot`, `etpmv`, `etbsv`, `yaxpy`, `yher` as unverified, not as losses. Verdicts come ONLY from the **warmed interleaved min-of-N wall-clock harness** (par/ob alternated run-by-run to cancel block drift; taskset-pinned, `OMP_WAIT_POLICY=passive`), read against a measured par-vs-par noise floor (~1.00 at OMP=1, ~1.004 at OMP=4). The five recently-fixed serial routines (`eynrm2` `erot` `yerot` `eaxpy` `yher2`) and the rewritten L3 set are all signed off there. See memory `project_x87_accumulator_spill` and `project_l3_fused_packer_pattern` for the per-routine interleaved results and commits.")
    out.append("")

    # 1. migrated drift sanity check.
    out.append("## Migrated-baseline drift across runs (sanity check)")
    out.append("")
    out.append("`max/min` ratio of the four `mig_*` readings per (routine,key,size). Closer to 1.00 ⇒ the migrated_ symbol's wall time is invariant w.r.t. OMP env / which binary it lived in.")
    out.append("")
    drifts = []
    for r in rows:
        mig = [pf(r[c]) for c in MIG_COLS]
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

    # 2. Per-routine median ns/call across all (key, size).
    out.append("## Per-routine medians (ns/call, smaller = faster) — across all (key, size)")
    out.append("")
    out.append(f"Columns: routine, then median ns/call for each of the 5 variants, then the par/ob wall-time ratios (**< 1.0 ⇒ parallel faster** — the firm bar is ≤ 1.0 in every cell), then the OMP=4/OMP=1 wall-time ratio for each C overlay (< 1.0 = threading helps; ~0.25 is ideal 4-core scaling).")
    out.append("")
    out.append(f"| routine | {EP1.label} | {EP4.label} | {P1.label} | {P4.label} | {MIG_LABEL} | par1/ob1 | par4/ob4 | ep4/ep1 | par4/par1 |")
    out.append( "|---------|--------------:|--------------:|--------------:|--------------:|--------------:|---------:|---------:|--------:|----------:|")
    for routine in sorted(by_routine):
        rr = by_routine[routine]
        cols = {v.tsv_col: [pf(r[v.tsv_col]) for r in rr] for v in SUBJECTS}
        cols = {k: [x for x in vs if x] for k, vs in cols.items()}
        mig = [v for v in (pf(r[MIGRATED_COL]) for r in rr) if v]
        if not (all(cols.values()) and mig):
            continue
        m = {k: median(vs) for k, vs in cols.items()}
        m_mig = median(mig)
        out.append(
            f"| {routine} | {m[EP1.tsv_col]:.1f} | {m[EP4.tsv_col]:.1f} "
            f"| {m[P1.tsv_col]:.1f} | {m[P4.tsv_col]:.1f} | {m_mig:.1f} "
            f"| {m[P1.tsv_col]/m[EP1.tsv_col]:.2f}× | {m[P4.tsv_col]/m[EP4.tsv_col]:.2f}× "
            f"| {m[EP4.tsv_col]/m[EP1.tsv_col]:.2f}× | {m[P4.tsv_col]/m[P1.tsv_col]:.2f}× |"
        )

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
    out.append("For each routine, the row at the largest N actually measured (ns/call, smaller = faster; par/ob ratio < 1.0 = parallel faster). Useful for cases where a serial-OK overlay regresses only at large N (or vice versa).")
    out.append("")
    out.append(f"| routine | key | N | {EP1.label} | {EP4.label} | {P1.label} | {P4.label} | {MIG_LABEL} | par1/ob1 | par4/ob4 | ep1/mig | par1/mig | par4/mig |")
    out.append( "|---------|-----|--:|--------------:|--------------:|--------------:|--------------:|--------------:|---------:|---------:|--------:|---------:|---------:|")
    for routine in sorted(big_n_routines):
        rr = big_n_routines[routine]
        n_max = max(n for n, _ in rr)
        # pick the (routine, key, N=n_max) row; if multiple keys at this N,
        # show the largest-N row with the lexicographically-first key.
        cands = [r for n, r in rr if n == n_max]
        cands.sort(key=lambda r: r["key"])
        r = cands[0]
        vals = {v.tsv_col: pf(r[v.tsv_col]) for v in SUBJECTS}
        mig = pf(r[MIGRATED_COL])
        if not (all(vals.values()) and mig):
            continue
        ep1, ep4 = vals[EP1.tsv_col], vals[EP4.tsv_col]
        p1,  p4  = vals[P1.tsv_col],  vals[P4.tsv_col]
        out.append(
            f"| {routine} | {r['key']} | {n_max} | {ep1:.1f} | {ep4:.1f} | {p1:.1f} | {p4:.1f} | {mig:.1f} "
            f"| {p1/ep1:.2f}× | {p4/ep4:.2f}× "
            f"| {ep1/mig:.2f}× | {p1/mig:.2f}× | {p4/mig:.2f}× |"
        )
    out.append("")

    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
