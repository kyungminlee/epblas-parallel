#!/usr/bin/env python3
"""Generate kernel-isolated C perf harnesses for every parallel-BLAS overlay routine.

One perf_<routine>.{c,cpp} per (target, routine) under
tests/epblas-parallel/perf/target_<target>/. CMake in
tests/epblas-parallel/CMakeLists.txt globs perf_*.{c,cpp} and links each
with -ffunction-sections / --gc-sections so overlay's and migrated's
symbols stay within a few KB of each other (no iTLB-spread artifact —
see findings doc Addendum 14).

Implementation lives in scripts/_perf_harness/ — one module per BLAS level:

    core.py         TypeInfo + per-target preambles + catalogs + routine_shape
    emit_l1.py      L1 emitters (axpy, dot, asum/nrm2, scal, rot*, cabs1, ...)
    emit_l2_dense.py    L2 dense  (gemv, ger, symv/hemv, syr/her, trmv/trsv)
    emit_l2_banded.py   L2 banded (gbmv, sbmv/hbmv, tbmv/tbsv)
    emit_l2_packed.py   L2 packed (spr/hpr, spmv/hpmv, tpmv/tpsv)
    emit_l3.py      L3 emitters (gemm, symm/hemm, syrk/herk, ...)
    dispatch.py     shape → emitter dispatch table + emit_routine entry point

Run from repo root:
    python3 scripts/gen_perf_harnesses.py [--targets ...] [--routines ...]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _perf_harness import CATALOG, GEN_SENTINEL, PERF_DIR, TYPES, emit_routine, routine_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="kind10,kind16,multifloats")
    ap.add_argument("--routines", default="")
    args = ap.parse_args()
    only = set(args.routines.split(",")) if args.routines else None

    written = 0
    skipped = []
    for tgt in args.targets.split(","):
        ti = TYPES[tgt]
        outdir = PERF_DIR / f"target_{tgt}"
        outdir.mkdir(parents=True, exist_ok=True)
        for name in CATALOG[tgt]:
            if only and name not in only:
                continue
            outpath = outdir / f"perf_{name}.{ti.file_ext}"
            # Skip files that don't contain our sentinel — they're
            # hand-written and the generator must not clobber them.
            if outpath.exists() and GEN_SENTINEL not in outpath.read_text():
                continue
            src = emit_routine(name, ti)
            outpath.write_text(src)
            written += 1
            if "not implemented" in src:
                skipped.append((tgt, name))
    print(f"wrote {written} files; {len(skipped)} unsupported")
    if skipped:
        shapes = Counter()
        for tgt, name in skipped:
            s, _ = routine_shape(name)
            shapes[s] += 1
        for s, n in sorted(shapes.items(), key=lambda x: -x[1]):
            print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
