#!/usr/bin/env python3
"""Aggregate a gap5_raw.tsv sweep into per-(routine,key,size) timing stats.

Accuracy protocol (reports/cmp5/BENCH_PROTOCOL.md):
  * Point estimate  : ratio-of-mins  = min_rep(par) / min_rep(ob).  The min over
    reps is the maximum-likelihood floor under one-sided additive noise.
  * Robust estimate : median of the PER-REP paired ratios par_i/ob_i, with a
    bootstrap 95% CI. Pairing by rep cancels CPU-frequency drift (par_i and ob_i
    run back-to-back at ~the same frequency), so this is trustworthy even when
    the governor/turbo couldn't be pinned. A cell is flagged SLOW only when the
    CI lower bound exceeds 1.0 (genuinely slower, not noise) — this replaces the
    old fixed 1.03 fudge factor.
  * If ratio-of-mins and median-per-rep-ratio disagree materially, residual
    frequency drift or bimodality is present and the cell is marked ~DRIFT.

Raw row: run_id omp routine key size iters subject_ns migrated_ns [rep]
(8-col legacy files without `rep` are accepted: reps are assigned by appearance
order per leg, which still pairs correctly when legs were emitted round-robin.)
"""
import sys, collections, statistics


def load(paths, series=None, mig=None):
    """Parse one or more gap5_raw.tsv files into
       series[(routine,key,size)][leg][rep] = min_ns  and  mig[cell] = min_ns.
    Later files override a routine's rows entirely (newest-fix-wins): if a
    routine appears in more than one file, only its last file's rows are kept."""
    if isinstance(paths, str):
        paths = [paths]
    if series is None:
        series = collections.defaultdict(lambda: collections.defaultdict(dict))
    if mig is None:
        mig = collections.defaultdict(lambda: float("inf"))
    owner = {}
    for path in paths:
        with open(path) as f:
            next(f)
            for line in f:
                rt = line.split("\t", 3)
                if len(rt) >= 3:
                    owner[rt[2]] = path
    seqctr = collections.defaultdict(int)
    for path in paths:
        with open(path) as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 8:
                    continue
                run_id, omp, routine, key, size, iters, subj, migns = p[:8]
                if owner.get(routine) != path:
                    continue
                try:
                    v = float(subj); k = (routine, key, int(size))
                except ValueError:
                    continue
                if len(p) >= 9 and p[8] != "":
                    rep = p[8]
                else:
                    sk = (k, run_id); seqctr[sk] += 1; rep = str(seqctr[sk])
                d = series[k][run_id]
                if rep not in d or v < d[rep]:
                    d[rep] = v
                try:
                    m = float(migns)
                    if m < mig[k]:
                        mig[k] = m
                except ValueError:
                    pass
    return series, mig


def boot_ci(ratios, iters=2000, lo=2.5, hi=97.5, seed=1234567):
    """Percentile bootstrap CI of the median of a small ratio sample.
    Deterministic LCG so runs are reproducible (no Math.random dependency)."""
    n = len(ratios)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (ratios[0], ratios[0])
    s = seed & 0xFFFFFFFF
    meds = []
    for _ in range(iters):
        samp = []
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            samp.append(ratios[s % n])
        meds.append(statistics.median(samp))
    meds.sort()
    def pct(q):
        idx = q / 100.0 * (len(meds) - 1)
        i = int(idx); fr = idx - i
        return meds[i] if i + 1 >= len(meds) else meds[i] * (1 - fr) + meds[i + 1] * fr
    return (pct(lo), pct(hi))


def cell_stats(s):
    """Return dict of stats for one cell's per-leg rep-series."""
    def mn(leg):
        d = s.get(leg, {})
        return min(d.values()) if d else float("nan")
    def paired_ratios(num, den):
        a, b = s.get(num, {}), s.get(den, {})
        reps = sorted(set(a) & set(b), key=lambda r: (len(r), r))
        return [a[r] / b[r] for r in reps if b[r] > 0]
    out = {"p1": mn("par-omp1"), "p4": mn("par-omp4"),
           "o1": mn("ob-omp1"), "o4": mn("ob-omp4")}
    for name, num, den in (("r1", "par-omp1", "ob-omp1"),
                           ("r4", "par-omp4", "ob-omp4")):
        pr = paired_ratios(num, den)
        out[name + "_ratios"] = pr
        out[name + "_med"] = statistics.median(pr) if pr else float("nan")
        out[name + "_ci"] = boot_ci(pr)
    return out


def worst_key(item):
    _, st = item
    cands = [c for c in (st["r1_med"], st["r4_med"]) if c == c]
    return max(cands) if cands else 0.0


def cell_flags(st):
    """Return (flag_list, is_slow) for one cell's stats."""
    rm1 = st["p1"] / st["o1"] if st["o1"] == st["o1"] and st["o1"] else float("nan")
    rm4 = st["p4"] / st["o4"] if st["o4"] == st["o4"] and st["o4"] else float("nan")
    flags = []
    for tag, med, ci, rm in (("p1", st["r1_med"], st["r1_ci"], rm1),
                             ("p4", st["r4_med"], st["r4_ci"], rm4)):
        lo, _ = ci
        if lo == lo and lo > 1.0:          # CI excludes 1.0 → genuinely slower
            flags.append(f"{tag}>ob")
        if med == med and rm == rm and abs(med - rm) > 0.04:
            flags.append(f"{tag}~DRIFT")
    return flags, any(fl.endswith(">ob") for fl in flags), rm1, rm4


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gap5_raw.tsv"
    series, _ = load(path)
    rows = [(k, cell_stats(series[k])) for k in series]
    rows.sort(key=worst_key, reverse=True)
    print(f"{'routine':8} {'key':10} {'N':>5} | "
          f"{'p1/o1':>6} [{'ci95':^13}] {'p4/o4':>6} [{'ci95':^13}] | "
          f"{'mins:p1/o1':>10} {'p4/o4':>6} | FLAG")
    print("-" * 108)
    fails = 0
    for k, st in rows:
        routine, key, size = k
        flags, slow, rm1, rm4 = cell_flags(st)
        fails += slow
        c1lo, c1hi = st["r1_ci"]; c4lo, c4hi = st["r4_ci"]
        print(f"{routine:8} {key:10} {size:5d} | "
              f"{st['r1_med']:6.3f} [{c1lo:5.3f},{c1hi:5.3f}] "
              f"{st['r4_med']:6.3f} [{c4lo:5.3f},{c4hi:5.3f}] | "
              f"{rm1:10.3f} {rm4:6.3f} | {' '.join(flags)}")
    print("-" * 108)
    print(f"SLOW cells (per-rep ratio CI lower bound > 1.0): {fails}")


if __name__ == "__main__":
    main()
