#!/usr/bin/env python3
"""Generate reports/overlay-coverage.md — table of every BLAS overlay routine
across the three extended-precision targets, with algorithm summary,
error digits (best-effort from test_* binaries), GFLOP/s at OMP=1, and
speedup vs the migrated reference at OMP=1 and OMP=4.

Usage:
    scripts/blas_overlay_report.py
        [--stage-root /tmp/fm-bench]
        [--size 256] [--iters 3]
        [--targets kind10,kind16,multifloats]
        [--out reports/overlay-coverage.md]
        [--skip-build]
        [--skip-bench]

Missing data is rendered as the placeholder "—". The script never
fails on a single routine — bench / test failures are recorded as
placeholders so partial runs still produce a useful report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "epblas-parallel"

TARGETS = ["kind10", "kind16", "multifloats"]
LIB_PREFIX = {"kind10": "e", "kind16": "q", "multifloats": "m"}
PARALLEL_TGT = {"kind10": "eblas_parallel",
                "kind16": "qblas_parallel",
                "multifloats": "mblas_parallel"}

PLACEHOLDER = "—"


# --------------------------------------------------------------------------
# Source introspection
# --------------------------------------------------------------------------
def overlay_routines(target: str) -> list[str]:
    """Read CMakeLists.txt and return routine basenames in source order.

    Walks the file line-by-line tracking parenthesis depth so comments
    that contain '(' / ')' don't terminate the set() block early."""
    lines = (SRC / target / "CMakeLists.txt").read_text().splitlines()
    out = []
    in_set = False
    depth = 0
    for line in lines:
        if not in_set:
            if re.match(r"\s*set\s*\(\s*_kernels\b", line):
                in_set = True
                depth = line.count("(") - line.count(")")
                continue
            continue
        # Inside set(_kernels ...). Update depth using only NON-comment text.
        code = line.split("#", 1)[0]
        depth += code.count("(") - code.count(")")
        s = code.strip().strip("()").strip()
        if s and re.match(r"^[A-Za-z0-9_]+\.(c|cpp)$", s):
            out.append(re.sub(r"\.(c|cpp)$", "", s))
        if depth <= 0:
            break
    return out


def isa_tag(target: str, routine: str) -> str:
    """Return 'AVX2+FMA3', 'AVX2', or 'scalar' based on hand-coded intrinsics.

    Describes what the SOURCE explicitly writes, not what the compiler
    may emit on top of scalar code.

    - 'AVX2+FMA3' — source uses _mm256_fmadd_pd / _mm256_fmsub_pd or the
      MBLAS_SIMD_DD / WBLAS_SIMD_DD macros (which expand to FMA3 paths).
    - 'AVX2'      — source uses _mm256_ intrinsics but no FMA3 intrinsics.
    - 'scalar'    — no SIMD intrinsics in the source. kind10 / kind16
      always land here; multifloats sources without _mm256_ also land
      here regardless of whatever the optimizer auto-vectorizes.
    """
    ext = "cpp" if target == "multifloats" else "c"
    p = SRC / target / f"{routine}.{ext}"
    if not p.exists():
        return PLACEHOLDER
    if target in ("kind10", "kind16"):
        return "scalar"
    src = p.read_text()
    has_avx = bool(re.search(r"__m256|_mm256_|MBLAS_SIMD_DD|WBLAS_SIMD_DD", src))
    has_fma = bool(re.search(r"_mm256_fm(add|sub|nmadd|nmsub)_pd"
                             r"|_mm_fm(add|sub|nmadd|nmsub)_[sp]d"
                             r"|MBLAS_SIMD_DD|WBLAS_SIMD_DD", src))
    if has_fma:
        return "AVX2+FMA3"
    if has_avx:
        return "AVX2"
    return "scalar"


def algorithm_summary(target: str, routine: str) -> str:
    """First descriptive line of the overlay source's top comment."""
    ext = "cpp" if target == "multifloats" else "c"
    p = SRC / target / f"{routine}.{ext}"
    if not p.exists():
        return PLACEHOLDER
    txt = p.read_text()
    m = re.match(r"\s*/\*(.*?)\*/", txt, re.DOTALL)
    if not m:
        return PLACEHOLDER
    block = m.group(1)
    # Strip leading " * " markers
    lines = [re.sub(r"^\s*\*\s?", "", ln).rstrip()
             for ln in block.splitlines()]
    # Find first non-banner line after the "name — ..." opener; if the
    # opener already contains a dash-separated description, use it.
    head = next((ln for ln in lines if ln.strip()), "")
    if "—" in head or "-" in head:
        # "qgemmtr — kind16 real triangular GEMM update."
        parts = re.split(r"[—–-]", head, maxsplit=1)
        if len(parts) == 2:
            desc = parts[1].strip().rstrip(".")
            if desc:
                return desc
    # Fall back to the first non-empty paragraph
    para = []
    started = False
    for ln in lines[1:]:
        if not ln.strip():
            if started: break
            continue
        started = True
        para.append(ln.strip())
        if len(" ".join(para)) > 160: break
    return (" ".join(para).rstrip(".") or PLACEHOLDER)[:200]


# --------------------------------------------------------------------------
# Build / run benches
# --------------------------------------------------------------------------
@dataclass
class BenchRow:
    target: str
    routine: str
    algo: str
    isa: str = PLACEHOLDER
    digits: str = PLACEHOLDER         # min / med / max digits
    gflops_omp1: str = PLACEHOLDER
    speedup_omp1: str = PLACEHOLDER
    speedup_omp4: str = PLACEHOLDER


def build_dir(stage_root: Path, target: str) -> Path:
    return stage_root / f"stage-{target}" / "build"


def cmake_build(bdir: Path, target_name: str) -> bool:
    try:
        r = subprocess.run(
            ["cmake", "--build", str(bdir),
             "--target", target_name, "-j4"],
            capture_output=True, text=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


_BENCH_ROW = re.compile(
    r"^\s*(?:[A-Z]+\s+)*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)x\s*$")


def run_bench_one(bdir: Path, routine: str, size: int, iters: int,
                  omp: int) -> list[tuple[float, float]] | None:
    """Run the bench and parse stdout. Returns list of
    (epblas-parallel GF/s, migrated GF/s) rows, or None on failure."""
    exe = bdir / "tests" / "epblas-parallel" / f"bench_{routine}"
    if not exe.exists():
        return None
    out = bdir / f"bench_{routine}_omp{omp}.json"
    env = os.environ.copy()
    env.update({
        "BLAS_BENCH_SIZES": str(size),
        "BLAS_BENCH_ITERS": str(iters),
        "BLAS_BENCH_WARMUP": "1",
        "BLAS_BENCH_OUT": str(out),
        "OMP_NUM_THREADS": str(omp),
    })
    try:
        r = subprocess.run([str(exe)], env=env, capture_output=True,
                           text=True, timeout=600)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    rows = []
    for ln in r.stdout.splitlines():
        m = _BENCH_ROW.match(ln)
        if m:
            try:
                rows.append((float(m.group(2)), float(m.group(3))))
            except ValueError:
                pass
    return rows or None


def summarize_bench(rows) -> tuple[float, float] | None:
    """Return (median epblas-parallel GF/s, median speedup epblas-parallel/migrated)."""
    if not rows:
        return None
    ov = [o for o, _ in rows]
    sp = [o / m for o, m in rows if m > 0]
    if not ov or not sp:
        return None
    return statistics.median(ov), statistics.median(sp)


# --------------------------------------------------------------------------
# Fuzz (per-routine consistency fuzzers — emit ERR_STATS line)
# --------------------------------------------------------------------------
import math

_ERR_STATS = re.compile(
    r"ERR_STATS:\s*min=\s*([\d.eE+-]+)\s+med=\s*([\d.eE+-]+)\s+max=\s*([\d.eE+-]+)\s+n=\s*(\d+)"
)


def _digits(err: float) -> str:
    if err <= 0.0:
        return ">32"
    d = -math.log10(err)
    return f"{d:.1f}"


def run_fuzz(bdir: Path, routine: str) -> str:
    """Run fuzz binary, parse ERR_STATS, return 'min/med/max' as digits."""
    exe = bdir / "tests" / "epblas-parallel" / f"fuzz_{routine}"
    if not exe.exists():
        return PLACEHOLDER
    try:
        r = subprocess.run([str(exe)], capture_output=True, text=True,
                           timeout=300)
    except Exception:
        return PLACEHOLDER
    if r.returncode != 0:
        # Fuzz failures (tol or sentinel) cause exit 1 — still useful
        # if ERR_STATS line is present.
        pass
    m = _ERR_STATS.search(r.stdout)
    if not m:
        return PLACEHOLDER
    vmin, vmed, vmax = float(m.group(1)), float(m.group(2)), float(m.group(3))
    # Higher digits = better, so swap min/max for digit reporting.
    return f"{_digits(vmax)} / {_digits(vmed)} / {_digits(vmin)}"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-root", default="/tmp/fm-bench")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--out", default="reports/overlay-coverage.md")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-fuzz", action="store_true")
    args = ap.parse_args()

    stage_root = Path(args.stage_root)
    targets = args.targets.split(",")

    rows: list[BenchRow] = []

    for target in targets:
        print(f"[{target}] enumerating overlay routines…", file=sys.stderr)
        bdir = build_dir(stage_root, target)
        routines = overlay_routines(target)
        if not args.skip_build and bdir.exists():
            cmake_build(bdir, PARALLEL_TGT[target])

        # Bench binaries are all under one cmake target named 'bench_<r>'.
        bench_targets = []
        for r in routines:
            shim = next(stage_root.glob(f"stage-{target}/tests/epblas-parallel/bench/target_*/bench_{r}.fypp"), None)
            if shim is not None:
                bench_targets.append(f"bench_{r}")
        if bench_targets and not args.skip_build:
            print(f"[{target}] building {len(bench_targets)} bench binaries…", file=sys.stderr)
            # Build in chunks to keep cmd lines sane
            for i in range(0, len(bench_targets), 30):
                chunk = bench_targets[i:i+30]
                subprocess.run(["cmake", "--build", str(bdir),
                                "--target", *chunk, "-j4"],
                               capture_output=True, timeout=1200)

        for r in routines:
            algo = algorithm_summary(target, r)
            row = BenchRow(target=target, routine=r, algo=algo,
                           isa=isa_tag(target, r))

            if not args.skip_bench:
                d1 = run_bench_one(bdir, r, args.size, args.iters, omp=1)
                d4 = run_bench_one(bdir, r, args.size, args.iters, omp=4)
                s1 = summarize_bench(d1) if d1 else None
                s4 = summarize_bench(d4) if d4 else None
                if s1:
                    row.gflops_omp1 = f"{s1[0]:.3f}"
                    row.speedup_omp1 = f"{s1[1]:.2f}×"
                if s4:
                    row.speedup_omp4 = f"{s4[1]:.2f}×"

            if not args.skip_fuzz:
                row.digits = run_fuzz(bdir, r)

            rows.append(row)
            print(f"  {target:<11} {r:<10} gf1={row.gflops_omp1:<8} "
                  f"sp1={row.speedup_omp1:<7} sp4={row.speedup_omp4:<7} "
                  f"err={row.digits}", file=sys.stderr)

    # Emit markdown
    out_lines = [
        "# epblas-parallel overlay coverage report",
        "",
        f"Auto-generated by `scripts/blas_overlay_report.py` from `src/epblas-parallel/`. "
        f"Size = {args.size}, iters = {args.iters}, OMP = 1 / 4. Speedup = "
        "`epblas-parallel GFLOP/s ÷ migrated-Fortran GFLOP/s` (>1 = epblas-parallel wins). "
        "Error digits are "
        "`min / median / max` across the test binary's cases (best-effort; "
        "many test programs cover only a handful of (uplo, trans) combinations).",
        "",
        "Scope: epblas-parallel overlay only. (epblas-openblas is a separate overlay, "
        "currently kind10-only; see `reports/cmp5/` for epblas-openblas vs epblas-parallel.)",
        "",
        f"Placeholder `{PLACEHOLDER}` = data unavailable "
        "(no bench shim, build failure, or no matching fuzz program).",
        "",
        "**Coverage:**",
        "",
    ]
    for target in targets:
        sub = [r for r in rows if r.target == target]
        nb = sum(1 for r in sub if r.gflops_omp1 != PLACEHOLDER)
        nf = sum(1 for r in sub if r.digits != PLACEHOLDER)
        out_lines.append(
            f"- {target}: {len(sub)} epblas-parallel routines, "
            f"{nb} with bench data, {nf} with fuzz data"
        )
    out_lines.append("")
    for target in targets:
        sub = [r for r in rows if r.target == target]
        if not sub: continue
        out_lines += [f"## {target}", "",
                      "| routine | algorithm | ISA | err digits (min/med/max) | epblas-parallel GFLOP/s (OMP=1) | speedup vs migrated (OMP=1) | speedup vs migrated (OMP=4) |",
                      "|---------|-----------|-----|---------------------------|-----------------------------:|---------------------------:|---------------------------:|"]
        for r in sub:
            algo = r.algo.replace("|", "\\|")[:120]
            out_lines.append(
                f"| `{r.routine}` | {algo} | {r.isa} | {r.digits} | "
                f"{r.gflops_omp1} | {r.speedup_omp1} | {r.speedup_omp4} |"
            )
        out_lines.append("")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
