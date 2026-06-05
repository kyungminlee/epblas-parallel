#!/usr/bin/env python3
"""Aggregate the 4-variant perf sweep into a comparison table.

Input columns (cmp5_raw.tsv):
    run_id run_binary omp taskset routine key size iters subject_ns migrated_ns

`subject_ns` is the bare wall time (ns/call, smaller = faster) of the C overlay
routine under test in this row; which overlay (epopenblas or parallel-blas) is
read from run_id. The same perf_<r>.c source is linked into both ep_perf_<r>
(against epopenblas) and perf_<r> (against parallel-blas), so within each row we
get one subject ns reading and one migrated ns reading (the migrated_
Fortran-reference symbol's timing inside that same binary).

Output:
    cmp5.tsv  — wide table with columns (all values ns/call, smaller = faster)
        routine key size epopenblas-omp1 epopenblas-omp4
                  parallel-blas-omp1 parallel-blas-omp4 migrated-serial
                  mig_par_omp1 mig_par_omp4 mig_ep_omp1 mig_ep_omp4

`migrated-serial` is the migrated_ns from the parallel-blas-omp1 run.
The mig_* columns expose all four migrated readings so we can spot-check
that the migrated_ symbol's timing is stable across runs (it should be —
`_serial` contains no OpenMP).
"""
import csv
from collections import defaultdict
from pathlib import Path

from columns import ALL_TSV_COLS, MIG_COLS, MIGRATED_COL, RUNID_TO_MIG, SUBJECTS


RAW = Path(__file__).parent / "cmp5_raw.tsv"
OUT = Path(__file__).parent / "cmp5.tsv"


def main():
    subject = defaultdict(dict)   # (routine, key, size) → {variant: ns}
    migrated = defaultdict(dict)  # (routine, key, size) → {mig_*: ns}
    with RAW.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key = (row["routine"], row["key"], int(row["size"]))
            run = row["run_id"]
            # MIN-merge across repeated raw rows (the interleaved sweep emits one
            # row per (cell, run_id) per rep; min-of-N is the authoritative
            # reduction — it discards transient contention/drift, keeping the
            # warmest reading). With a single rep this is a no-op (min of one).
            s = float(row["subject_ns"])
            mg = float(row["migrated_ns"])
            mc = RUNID_TO_MIG[run]
            if run in subject[key]:
                subject[key][run] = min(subject[key][run], s)
            else:
                subject[key][run] = s
            if mc in migrated[key]:
                migrated[key][mc] = min(migrated[key][mc], mg)
            else:
                migrated[key][mc] = mg

    rows = []
    for k in sorted(subject):
        routine, ks, size = k
        o = subject[k]
        m = migrated[k]
        row = {"routine": routine, "key": ks, "size": size}
        for v in SUBJECTS:
            row[v.tsv_col] = o.get(v.tsv_col, "")
        row[MIGRATED_COL] = m.get("mig_par_omp1", "")
        for c in MIG_COLS:
            row[c] = m.get(c, "")
        rows.append(row)

    with OUT.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(ALL_TSV_COLS), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.1f}" if isinstance(r[c], float) else r[c])
                        for c in ALL_TSV_COLS})
    print(f"wrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
