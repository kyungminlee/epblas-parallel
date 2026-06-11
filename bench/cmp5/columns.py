"""Canonical cmp5.tsv column names + display vocabulary.

Single source of truth for every report/script that reads cmp5.tsv. Adding a
new sweep variant means editing this file *only*; downstream consumers
(aggregate, summarize, par_wins, the CHECKLIST scripts) import from here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    tsv_col: str    # column name in cmp5.tsv (also the run_id in cmp5_raw.tsv)
    label:   str    # user-facing display name
    mig_col: str    # the mig_* sanity column written from this run


# Order matters: this is the order used in cmp5.tsv subject columns and in
# every report table header. Subjects are ep-first / par-second by historical
# convention.
SUBJECTS = (
    Variant("epopenblas-omp1",    "openblas-omp1",  "mig_ep_omp1"),
    Variant("epopenblas-omp4",    "openblas-omp4",  "mig_ep_omp4"),
    Variant("parallel-blas-omp1", "parallel-omp1",  "mig_par_omp1"),
    Variant("parallel-blas-omp4", "parallel-omp4",  "mig_par_omp4"),
)

# The fifth "variant" — the migrated Fortran reference. Same numeric source as
# the mig_par_omp1 sanity column, surfaced as its own first-class column.
MIGRATED_COL   = "migrated-serial"
MIGRATED_LABEL = "eplinalg-omp1"

# mig_* columns in the order cmp5.tsv writes them. Preserved as-is from the
# original aggregator: par-first / ep-second (which intentionally does NOT
# match the subject order above). Don't reorder without re-aggregating every
# downstream report.
MIG_COLS = ("mig_par_omp1", "mig_par_omp4", "mig_ep_omp1", "mig_ep_omp4")

# Full cmp5.tsv column list in write order.
ALL_TSV_COLS = (
    "routine", "key", "size",
    *(v.tsv_col for v in SUBJECTS),
    MIGRATED_COL,
    *MIG_COLS,
)

# cmp5_raw.tsv run_id → corresponding mig_* column.
RUNID_TO_MIG = {v.tsv_col: v.mig_col for v in SUBJECTS}
