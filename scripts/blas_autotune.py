#!/usr/bin/env python3
"""Block-size autotune for parallel BLAS gemm kernels.

Sweeps EBLAS/QBLAS/MBLAS_{MC,KC,NC} env vars over a grid, runs the
target's bench_*gemm executable at a chosen problem size, records
GFLOP/s, and writes a per-target JSON report with the recommended
defaults (best median GFLOP/s).

Usage:
    python scripts/blas_autotune.py <build-dir> --target {kind10|kind16|multifloats}
        [--routine gemm] [--size 1024] [--threads 4]

Examples:
    python scripts/blas_autotune.py /path/to/epparablas/build --target kind10
    python scripts/blas_autotune.py /path/to/epparablas/build --target kind16 --size 512
    python scripts/blas_autotune.py /path/to/epparablas/build --target multifloats --size 256

Output: <build-dir>/reports/autotune_<target>_<routine>.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PREFIXES: Dict[str, Tuple[str, str]] = {
    # target → (real_prefix, env_prefix)
    "kind10":      ("e", "EBLAS"),
    "kind16":      ("q", "QBLAS"),
    "multifloats": ("m", "MBLAS"),
}

# Grid of (MC, KC, NC) values to sweep. Picked to cover L1/L2/L3
# fits for typical x86 caches without exploding the runtime.
MC_GRID = [32, 64, 96, 128, 192]
KC_GRID = [64, 128, 192, 256]
NC_GRID = [128, 256, 384, 512]


def run_one(bench: Path, env_prefix: str, mc: int, kc: int, nc: int,
            size: int, threads: int, iters: int) -> float | None:
    """Run bench at (mc, kc, nc); return subject (parallel-blas) GFLOP/s (or None on failure)."""
    env = os.environ.copy()
    env[f"{env_prefix}_MC"] = str(mc)
    env[f"{env_prefix}_KC"] = str(kc)
    env[f"{env_prefix}_NC"] = str(nc)
    env["OMP_NUM_THREADS"] = str(threads)
    env["BLAS_BENCH_SIZES"] = str(size)
    env["BLAS_BENCH_ITERS"] = str(iters)
    env["BLAS_BENCH_WARMUP"] = "1"
    out_json = bench.parent / f".tune_{env_prefix}_{mc}_{kc}_{nc}.json"
    env["BLAS_BENCH_OUT"] = str(out_json)
    try:
        subprocess.run([str(bench)], env=env, check=True,
                       capture_output=True, timeout=600)
        data = json.loads(out_json.read_text())
        return float(data["results"][0]["gflops_subject"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"  ! {mc}/{kc}/{nc}: {e}", file=sys.stderr)
        return None
    finally:
        out_json.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("build_dir", type=Path,
                   help="CMake build dir (e.g. /path/to/epparablas/build)")
    p.add_argument("--target", required=True, choices=list(PREFIXES))
    p.add_argument("--routine", default="gemm",
                   help="routine family to tune (default: gemm)")
    p.add_argument("--size", type=int, default=1024,
                   help="problem size (default: 1024)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                   help="OMP_NUM_THREADS (default: detected)")
    p.add_argument("--iters", type=int, default=2,
                   help="timed iterations per combo (default: 2; keeps "
                        "total runtime bounded over the full grid)")
    args = p.parse_args()

    real_prefix, env_prefix = PREFIXES[args.target]
    bench_name = f"bench_{real_prefix}{args.routine}"
    bench = args.build_dir / "tests" / "blas_parallel" / bench_name
    if not bench.exists():
        print(f"error: {bench} not found — build it first "
              f"(`cmake --build {args.build_dir} --target {bench_name}`)",
              file=sys.stderr)
        return 1

    grid = list(itertools.product(MC_GRID, KC_GRID, NC_GRID))
    print(f"Autotuning {bench_name} @ N={args.size}, "
          f"OMP_NUM_THREADS={args.threads}, "
          f"{len(grid)} combos × {args.iters} iters each")

    results: List[Dict[str, float | int]] = []
    best = (-1.0, (0, 0, 0))
    for i, (mc, kc, nc) in enumerate(grid, 1):
        g = run_one(bench, env_prefix, mc, kc, nc,
                    args.size, args.threads, args.iters)
        if g is None:
            continue
        results.append({"mc": mc, "kc": kc, "nc": nc, "gflops": g})
        marker = ""
        if g > best[0]:
            best = (g, (mc, kc, nc))
            marker = "  <-- best so far"
        print(f"  [{i:3d}/{len(grid)}] {mc:3d}/{kc:3d}/{nc:4d}: "
              f"{g:9.4f} GFLOP/s{marker}")

    report = {
        "target": args.target,
        "routine": f"{real_prefix}{args.routine}",
        "size": args.size,
        "threads": args.threads,
        "iters": args.iters,
        "best": {"mc": best[1][0], "kc": best[1][1], "nc": best[1][2],
                 "gflops": best[0]},
        "env_vars": {
            f"{env_prefix}_MC": best[1][0],
            f"{env_prefix}_KC": best[1][1],
            f"{env_prefix}_NC": best[1][2],
        },
        "grid": sorted(results, key=lambda r: -r["gflops"]),
    }

    out_dir = args.build_dir / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"autotune_{args.target}_{real_prefix}{args.routine}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nBest: MC={best[1][0]} KC={best[1][1]} NC={best[1][2]} "
          f"→ {best[0]:.4f} GFLOP/s")
    print(f"Wrote {out_path}")
    print(f"To use: export "
          f"{env_prefix}_MC={best[1][0]} "
          f"{env_prefix}_KC={best[1][1]} "
          f"{env_prefix}_NC={best[1][2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
