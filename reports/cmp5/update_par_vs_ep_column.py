#!/usr/bin/env python3
"""Update the existing `par>ep (omp1/omp4)` column in src/epblas-openblas/CHECKLIST.md.

Unlike `insert_par_vs_ep_column.py` (which adds a brand-new column), this
script REPLACES the cell contents in the column, leaving everything else
untouched. Run it after re-running the cmp5 sweep + `aggregate.py`.

For each routine row, the cell becomes:
- "—"            if neither OMP=1 nor OMP=4 par/ep ratio crosses 10%
- "<v>× / —"     OMP=1 only
- "— / <v>×"     OMP=4 only
- "<v1>× / <v2>×" both directions cross 10%
- "n/d"          routine has no cmp5 rows (timed out or no perf binary)
- "n/a"          routine declared no-bench (enrm2, eynrm2, ecabs1)
- ""             status==todo (no data expected yet)
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMP = ROOT / "reports/cmp5/cmp5.tsv"
CHK = ROOT / "src/epblas-openblas/CHECKLIST.md"

THRESH = 1.10
NA_ROUTINES = {"enrm2", "eynrm2", "ecabs1"}
HEADER_TOKEN = "par>ep"


def pf(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load_ratios():
    omp1, omp4, seen = defaultdict(float), defaultdict(float), set()
    for r in csv.DictReader(CMP.open(), delimiter="\t"):
        rt = r["routine"]; seen.add(rt)
        ep1 = pf(r["epopenblas-omp1"]); p1 = pf(r["parallel-blas-omp1"])
        ep4 = pf(r["epopenblas-omp4"]); p4 = pf(r["parallel-blas-omp4"])
        if ep1 and p1: omp1[rt] = max(omp1[rt], p1 / ep1)
        if ep4 and p4: omp4[rt] = max(omp4[rt], p4 / ep4)
    return omp1, omp4, seen


def cell_for(rt, omp1, omp4, seen):
    if rt not in seen: return "n/d"
    m1, m4 = omp1.get(rt, 0.0), omp4.get(rt, 0.0)
    c1 = f"{m1:.2f}×" if m1 >= THRESH else "—"
    c4 = f"{m4:.2f}×" if m4 >= THRESH else "—"
    return f"{c1} / {c4}" if (m1 >= THRESH or m4 >= THRESH) else "—"


def find_par_ep_col(header_line):
    """Return zero-based index of par>ep cell, or None."""
    cells = header_line.split("|")
    for j, c in enumerate(cells):
        if HEADER_TOKEN in c:
            return j
    return None


def cell_width(header_line, col_idx):
    """Return length (without leading/trailing pipe) of cell in header row."""
    cells = header_line.split("|")
    return len(cells[col_idx])


def main():
    omp1, omp4, seen = load_ratios()
    # Empty / stale cmp5.tsv → seen is empty → every cell becomes "n/d".
    # Refuse to mutate; the user almost certainly forgot to re-run the
    # sweep before this script.
    if not seen:
        sys.exit(f"{CMP} has no rows — refusing to overwrite column "
                 f"(run run_cmp5.sh + aggregate.py first)")
    src = CHK.read_text().splitlines(keepends=False)
    out = []

    par_ep_col = None
    par_ep_width = None

    for line in src:
        # Track the active table's par>ep column index.
        if line.startswith("| target") and HEADER_TOKEN in line:
            par_ep_col = find_par_ep_col(line)
            par_ep_width = cell_width(line, par_ep_col) if par_ep_col is not None else None
            out.append(line); continue
        if line.startswith("|---") and par_ep_col is not None:
            out.append(line); continue

        # End-of-table sentinel: blank line or new heading.
        if par_ep_col is not None and (not line.startswith("|")):
            par_ep_col = None
            par_ep_width = None
            out.append(line); continue

        # Data row of the active table.
        if par_ep_col is not None and line.startswith("|"):
            cells = line.split("|")
            if len(cells) > par_ep_col:
                rt = cells[1].strip()
                status = cells[3].strip() if len(cells) > 3 else ""
                if rt and status in {"todo", "wip", "done"}:
                    if rt in NA_ROUTINES:
                        body = "n/a"
                    elif status == "todo":
                        body = ""
                    else:
                        body = cell_for(rt, omp1, omp4, seen)
                    # Preserve the leading space convention `| body  |`.
                    cells[par_ep_col] = f" {body:<{par_ep_width-2}} "
                    out.append("|".join(cells))
                    continue
        out.append(line)

    CHK.write_text("\n".join(out) + ("\n" if src and src[-1] == "" else "\n"))
    print(f"updated {CHK}")


if __name__ == "__main__":
    main()
