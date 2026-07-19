#!/usr/bin/env bash
# update_routine.sh — re-time ONE routine (or a few) and refresh the committed
# scoreboard, without re-running the multi-day full surface.
#
# This is the "subroutine-by-subroutine" update cycle. It works because the dual
# results are one file per routine (<rt>.omp{1,4}.txt) and both agg_dual.py and
# render_scoreboard.py are pure reductions over the results dir: re-timing one
# routine overwrites only that routine's two files; every other cell is carried
# over from the last sweep when the doc re-renders.
#
# Pipeline (delegates to run_dual.sh for the family-specific build/ns/gen/run):
#   1. run_dual.sh <fam> <routines> with OUT pinned to the SWEEP's results dir
#      (results_<fam>) so the new files land beside the full-surface data, and
#      NOAGG=1 (we render the markdown doc instead of the console board).
#      Archives are REBUILT by default (anti-stale, [[feedback_stale_archive_trap]]);
#      pass SKIP_BUILD=1 only when you know no par/ob/mig source changed.
#   2. render_scoreboard.py regenerates doc/dev/benchmark/results.md from ALL three
#      results_{m,e,q} dirs.
#
# Usage:
#   benchmark/dual/update_routine.sh <family> <routine>[,<routine>...]
#     family   : e | q | m
#     routines : comma list (e.g. etbsv  or  etbsv,etbmv)
# Env: REPS (default 40 — keep ≥40 for sub-2% verdicts), SKIP_BUILD=1 to reuse
#      archives, CORE1/CORE4 (pin cores; do NOT run two pinned sweeps at once),
#      NSDIR (bench tree, default workspace/files/gap5/nsbench — passed through
#      to run_dual.sh AND honored by render_scoreboard.py, so an overridden
#      update renders from the same tree it timed into).
#
# Raw data stays in the gitignored workspace; only the rendered .md is committed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FAM="${1:?usage: update_routine.sh <e|q|m> <routine[,routine...]>}"
ROUTINES_CSV="${2:?usage: update_routine.sh <e|q|m> <routine[,routine...]>}"

NSDIR="${NSDIR:-$ROOT/workspace/files/gap5/nsbench}"
RESULTS="$NSDIR/results_$FAM"   # the full-surface sweep's per-family dir
[[ -d "$RESULTS" ]] || echo "note: $RESULTS does not exist yet — creating (first cells for family $FAM)" >&2

echo "### re-timing $FAM: $ROUTINES_CSV  (reps=${REPS:-40}) -> $RESULTS ###"
NSDIR="$NSDIR" OUT="$RESULTS" NOAGG=1 REPS="${REPS:-40}" \
  "$HERE/run_dual.sh" "$FAM" "$ROUTINES_CSV"

echo "### refresh committed scoreboard ###"
NSDIR="$NSDIR" python3 "$HERE/render_scoreboard.py"
