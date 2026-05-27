#!/usr/bin/env python3
"""Aggregate reports/perf_sweep.tsv into a Markdown summary.

Per-routine: median (parallel-blas / migrated) ratio over all (key, size)
cells, plus min/max. Highlights cells with ratio < 0.95× (parallel-blas
slower than migrated) or > 1.10× (parallel-blas faster) — useful for
spotting where the C harness's honest measurement differs from prior
Fortran-bench reports.

Scope: parallel-blas overlay vs migrated Fortran reference (epopenblas not
covered by this sweep — see `reports/cmp5/` for that comparison).

Usage:
    scripts/aggregate_perf_sweep.py [--tsv reports/perf_sweep.tsv]
        [--out reports/perf_sweep.md]
"""
from __future__ import annotations
import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='reports/perf_sweep.tsv')
    ap.add_argument('--out', default='reports/perf_sweep.md')
    args = ap.parse_args()

    tsv = Path(args.tsv)
    if not tsv.exists():
        sys.exit(f'no such file: {tsv}')

    rows = []
    with tsv.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            try:
                r['size'] = int(r['size'])
                r['iters'] = int(r['iters'])
                r['parallel_blas_GFs'] = float(r['parallel_blas_GFs'])
                r['migrated_GFs'] = float(r['migrated_GFs'])
                r['ratio'] = float(r['ratio'])
            except (ValueError, KeyError) as e:
                print(f'skip bad row: {r} ({e})', file=sys.stderr)
                continue
            rows.append(r)

    if not rows:
        sys.exit('no rows parsed')

    # Group by (target, routine).
    by_routine = collections.defaultdict(list)
    for r in rows:
        by_routine[(r['target'], r['routine'])].append(r)

    out = []
    out.append('# parallel-blas overlay vs migrated Fortran — C perf harness sweep (OMP=1)')
    out.append('')
    out.append(f'Source: `{tsv}` ({len(rows)} cells, {len(by_routine)} routines)')
    out.append('')
    out.append('Ratio column: `parallel-blas GF/s ÷ migrated GF/s` (>1 = parallel-blas wins).')
    out.append('')
    out.append('Methodology:')
    out.append('- Kernel-isolated C harness with `-ffunction-sections -Wl,--gc-sections`')
    out.append('  (per findings doc Addendum 14 — collapses parallel-blas overlay\'s symbol')
    out.append('  footprint to a few KB so iTLB churn doesn\'t inflate the gap).')
    out.append('- `taskset -c 0 OMP_NUM_THREADS=1` on Intel i3-1315U P-core.')
    out.append('- Per-routine summary shows median ratio over all (key, size) cells.')
    out.append('')

    for tgt in ('kind10', 'kind16', 'multifloats'):
        tgt_routines = sorted(name for (t, name) in by_routine if t == tgt)
        if not tgt_routines:
            continue
        out.append(f'## target_{tgt}')
        out.append('')
        out.append('| routine | cells | median × | min × | max × | min cell | max cell |')
        out.append('|---------|------:|---------:|------:|------:|----------|----------|')
        for name in tgt_routines:
            cells = by_routine[(tgt, name)]
            ratios = [c['ratio'] for c in cells]
            med = statistics.median(ratios)
            mn = min(cells, key=lambda c: c['ratio'])
            mx = max(cells, key=lambda c: c['ratio'])
            mark = ''
            if med < 0.92:
                mark = ' ⚠'
            elif med > 1.10:
                mark = ' ⬆'
            out.append(
                f'| `{name}` | {len(cells)} | {med:.3f}{mark} | '
                f'{mn["ratio"]:.3f} | {mx["ratio"]:.3f} | '
                f'`{mn["key"]}`@{mn["size"]} | `{mx["key"]}`@{mx["size"]} |'
            )
        out.append('')

    Path(args.out).write_text('\n'.join(out) + '\n')
    print(f'wrote {args.out} ({len(by_routine)} routines)')

if __name__ == '__main__':
    main()
