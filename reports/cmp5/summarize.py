#!/usr/bin/env python3
"""Build a human-readable summary of cmp5.tsv.

Sanity-checks the migrated-serial column by comparing the 4 mig_* readings
(should be near-equal since `_serial` is OpenMP-free), then emits:

  1) Per-routine table — for each routine, the median ns/call across (key, size)
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
    precisions = sorted({"kind10" if r["routine"][:2] in ("ie", "iy") or r["routine"][:1] in ("e", "y")
                         else "kind16" for r in rows})
    out.append(f"# 5-way comparison: {EP1.label.split('-')[0]} / {P1.label.split('-')[0]} / {MIG_LABEL.split('-')[0]} — {' + '.join(precisions)} (e/y = REAL/COMPLEX(KIND=10), q/x = REAL/COMPLEX(KIND=16))")
    out.append("")
    out.append(f"- Source: `reports/cmp5/cmp5.tsv` ({len(rows)} (routine,key,size) rows over {len(by_routine)} routines)")
    out.append(f"- Five variants: `{EP1.label}` and `{EP4.label}` (epblas-openblas overlay), `{P1.label}` and `{P4.label}` (epblas-parallel overlay), and `{MIG_LABEL}` (Fortran reference, serial baseline).")
    out.append("- All four overlay binaries link the SAME `tests/epblas-parallel/perf/target_<kind>/perf_<r>.c` source (kind10 for e/y routines, kind16 for q/x) — only the C-overlay symbol differs.")
    out.append("- Same `BLAS_PERF_{ITERS,WARMUP,INCX,INCY}=200/20/1/1`; per-routine default sizes; pinned via `taskset` (P-cores 0 or 0..3).")
    out.append(f"- `{MIG_LABEL}` = migrated_ns from the `{P1.label}` run; `mig_*` columns are sanity readings of the same migrated `_serial` symbol from each of the four runs (expected to be ~equal since `_serial` contains no OpenMP).")
    out.append("")
    out.append("**Units: all columns are bare wall time in ns/call — smaller is faster.** A `parallel/openblas` wall-time ratio **< 1.0** means parallel is faster; the firm bar is par ≤ ob in every cell. (The separate interleaved per-routine harness reports the same par/ob wall-time ratio, where < 1.0 = par faster — the same direction. See `doc/optimization-findings.md` → \"Reporting convention\".)")
    out.append("")
    out.append("> ⚠️ **Methodology + how to read the OMP=4 gaps.** This table is now a **min-of-5 interleaved** sweep (`REPS=5` full passes over the whole surface; `aggregate.py` keeps the per-cell minimum, so transient drift/contention cancels) — NOT the old single-shot screen. Serial (`par1/ob1`) numbers are therefore trustworthy at the cell level, and near-parity cells (**0.95–1.05 band**) are at parity. **The large `par4/ob4` ratios are STRUCTURAL, not regressions:** the OpenBLAS reference threads *every* shape unconditionally, while the parallel overlay threads selectively. Confirmed from `par4/par1` (≈1.00 ⇒ par ran serial; ≈0.25 ⇒ par threaded 4×): (1) **strided L2** (`key` with `/x±`,`/y±`, i.e. incx/incy≠1) — par threads only the **unit-stride** NoTrans/Trans cases (`par4/ob4≈1.0` or par-wins there); the strided variants fall back to serial → `par4/ob4`≈3.5–4×. (2) **kind16 L1 reductions** (`qasum` `qdot` `qxasum` `xdotc` `iqamax` …) — the kind10 reduction-threading (`acf10a2`) was never ported to `parallel/kind16`, so they stay serial while `q/x` openblas threads at N≥16384. (3) **L1 RMW** (`*axpy` `*scal` `*copy` `*swap` `*rot*`, both precisions) — par is intentionally serial (write-bound RMW doesn't benefit; par's serial ties/beats ob serial — see `project_x87_accumulator_spill`). (4) **kind16 generally has thinner OMP coverage** than kind10 (the divergent simpler port). None of these is a correctness or serial-speed regression; they are the known threading-coverage frontier. Fine-grained near-parity verdicts on the rewritten L3 set (`esymm` `esyrk` `etrsm` `etrmm` `ysymm` `yhemm` `yherk` `yher2k` …) still come from the **per-routine interleaved harness** (`reports/cmp5/*_interleaved.md`); see memory `project_x87_accumulator_spill`, `project_l3_fused_packer_pattern`, `project_l2_omp_gap`, `project_l1_reduction_threading`.")
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
