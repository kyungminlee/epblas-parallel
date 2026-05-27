#!/usr/bin/env python3
"""Insert a `par>ep` column into src/epopenblas/CHECKLIST.md.

For each routine row in each table, look up the peak par-blas / epopenblas
ratio at OMP=1 and OMP=4 (from reports/cmp5/cmp5.tsv) and emit a cell of
the form `omp1 / omp4`, e.g. `1.51× / 2.15×`. If either ratio is below
the 1.10× noise floor it becomes "—"; if both are below it the whole
cell is "—". Routines that timed out (no rows in cmp5.tsv) get "n/d".
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/kyungminlee/code/fortran-migrator")
CMP = ROOT / "reports/cmp5/cmp5.tsv"
CHK = ROOT / "src/epopenblas/CHECKLIST.md"

THRESH = 1.10
NEW_HEADER = "par>ep (omp1/omp4)"
NEW_SEP_LEN = 23  # match the bench-omp[14] column width


def pf(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load_ratios():
    omp1_max = defaultdict(float)
    omp4_max = defaultdict(float)
    seen = set()
    for r in csv.DictReader(CMP.open(), delimiter="\t"):
        rt = r["routine"]
        seen.add(rt)
        ep1 = pf(r["epopenblas-omp1"]); p1 = pf(r["parallel-blas-omp1"])
        ep4 = pf(r["epopenblas-omp4"]); p4 = pf(r["parallel-blas-omp4"])
        if ep1 and p1:
            omp1_max[rt] = max(omp1_max[rt], p1 / ep1)
        if ep4 and p4:
            omp4_max[rt] = max(omp4_max[rt], p4 / ep4)
    return omp1_max, omp4_max, seen


def cell_for(rt, omp1_max, omp4_max, seen):
    if rt not in seen:
        return "n/d"
    m1 = omp1_max.get(rt, 0.0)
    m4 = omp4_max.get(rt, 0.0)
    c1 = f"{m1:.2f}×" if m1 >= THRESH else "—"
    c4 = f"{m4:.2f}×" if m4 >= THRESH else "—"
    return f"{c1} / {c4}" if (m1 >= THRESH or m4 >= THRESH) else "—"


# Routines with no cmp5 bench coverage by design (no OpenBLAS interface bench
# or scalar O(1) — same as the bench-omp1 column is empty / "n/a").
NA_ROUTINES = {"enrm2", "eynrm2", "ecabs1"}


def main():
    omp1_max, omp4_max, seen = load_ratios()

    src = CHK.read_text().splitlines(keepends=False)
    out = []

    i = 0
    while i < len(src):
        line = src[i]

        # Match a header row like `| target | ... | notes ... |`
        if line.startswith("| target") and "notes" in line:
            # Split on pipes and insert column before notes.
            cells = line.split("|")
            # cells = ["", " target  ", " openblas source ...", " ...", ..., " notes ... ", ""]
            notes_idx = next(j for j, c in enumerate(cells) if c.strip().startswith("notes"))
            new_cell = f" {NEW_HEADER:<{NEW_SEP_LEN-2}} "
            cells.insert(notes_idx, new_cell)
            out.append("|".join(cells))
            i += 1
            # Next line MUST be the separator. Inject a matching separator cell.
            sep_line = src[i]
            assert sep_line.lstrip().startswith("|---"), f"expected separator, got {sep_line!r}"
            sep_cells = sep_line.split("|")
            sep_cells.insert(notes_idx, "-" * NEW_SEP_LEN)
            out.append("|".join(sep_cells))
            i += 1
            continue

        # Match a data row: starts with `| <name> |` and the second cell looks
        # like a routine name (alpha only, length 4-8). Only act when the
        # routine is one we have a row for.
        if line.startswith("|"):
            cells = line.split("|")
            if len(cells) >= 9:
                rt = cells[1].strip()
                # Distinguish data rows from anything else by checking that
                # cells[3] (status) is one of the known values.
                status_cell = cells[3].strip() if len(cells) > 3 else ""
                if status_cell in {"todo", "wip", "done"} and rt:
                    if rt in NA_ROUTINES:
                        body = "n/a"
                    elif status_cell == "todo":
                        body = ""
                    else:
                        body = cell_for(rt, omp1_max, omp4_max, seen)
                    new_cell = f" {body:<{NEW_SEP_LEN-2}} "
                    # Insert before the notes cell. Structure:
                    # ["", "target", ..., "bench-omp4", "notes", ""].
                    # notes is at len(cells)-2; insert there to push notes right.
                    cells.insert(len(cells) - 2, new_cell)
                    out.append("|".join(cells))
                    i += 1
                    continue

        out.append(line)
        i += 1

    CHK.write_text("\n".join(out) + ("\n" if src and not src[-1] else ""))
    print(f"wrote {CHK}")


if __name__ == "__main__":
    main()
