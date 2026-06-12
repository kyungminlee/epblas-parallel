#!/usr/bin/env python3
"""Generate dual-link (option-2) perf drivers for the parallel-BLAS overlay.

One dual_<routine>.{c,cpp} per routine under bench/dual/drivers/. Each driver
links the namespaced par_/ob_/mig_ archives (built by bench/dual/nsbuild.sh)
into a SINGLE process and times all three legs interleaved per rep on shared
buffers — killing the cross-process layout/frequency bias the legacy
separate-binary cmp5 harness suffered, while keeping the real linked routine
(dispatch, thresholds, PLT) and the threaded path.

Implementation lives in scripts/_perf_harness/:
    dual.py          Family configs + render() (the uniform 3-leg timing loop)
    dual_shapes.py   per-shape Spec builders (signature/setup/call/sweep)

Run from repo root:
    python3 scripts/gen_dual_harnesses.py [--routines a,b,c]

With no --routines, generates every CATALOG routine whose shape is supported;
unsupported shapes are listed at the end.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _perf_harness.core import CATALOG, routine_shape
from _perf_harness.dual import family_for, render
from _perf_harness.dual_shapes import build_spec

OUTDIR = Path(__file__).resolve().parent.parent / "bench" / "dual" / "drivers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routines", default="",
                    help="comma-separated routine basenames (default: whole catalog)")
    ap.add_argument("--family", default="",
                    help="restrict to one family key: e | q | m (default: all)")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--list", action="store_true",
                    help="print emitted routine names to stdout (summary to stderr)")
    args = ap.parse_args()
    only = set(args.routines.split(",")) if args.routines else None
    fam_key = args.family.strip() or None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # De-dup across families (a routine name is unique to one family anyway).
    names = []
    for fam_routines in CATALOG.values():
        for name in fam_routines:
            if only and name not in only:
                continue
            if fam_key and family_for(name).key != fam_key:
                continue
            names.append(name)

    emitted = []
    skipped = []
    for name in names:
        built = build_spec(name)
        if built is None:
            skipped.append(name)
            continue
        is_c, ret, sig, spec = built
        fam = family_for(name)
        src = render(name, fam, is_c, ret, sig, spec)
        outpath = outdir / f"dual_{name}.{fam.ext}"
        outpath.write_text(src)
        emitted.append(name)

    log = sys.stderr if args.list else sys.stdout
    print(f"wrote {len(emitted)} drivers to {outdir}; {len(skipped)} unsupported", file=log)
    if skipped:
        shapes = Counter()
        for name in skipped:
            s, _ = routine_shape(name)
            shapes[s] += 1
        for s, n in sorted(shapes.items(), key=lambda x: -x[1]):
            print(f"  {s}: {n}", file=log)
    if args.list:
        print(" ".join(emitted))


if __name__ == "__main__":
    main()
