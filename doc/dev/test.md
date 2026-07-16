# Test

Two kinds of testing: **correctness** (the fuzz gate — run after every change)
and **performance** (the dual-link perf harness — run when you touch a hot
path).

## Correctness: the fuzz gate

Each fuzz driver A/Bs the overlay against eplinalg's migrated netlib baseline
over randomized dims/args and fails on the first mismatch. This is the
correctness gate; the serial and threaded paths must both match the baseline
bit-for-bit (identical summation order).

```bash
cmake --workflow --preset fuzz                    # configure + build + gate
ctest --preset fuzz                               # re-run the gate, no rebuild
BLAS_FUZZ_SEED=42 ctest --test-dir build -L fuzz  # one fixed seed
BLAS_FUZZ_SEED=42 ctest --test-dir build -L fuzz -R qtrmm   # one routine
```

| Env | Default | Effect |
|---|---|---|
| `BLAS_FUZZ_SEED` | time-seeded | fix the RNG for a reproducible run. **Always set it when reproducing a failure.** |
| `BLAS_FUZZ_CASES` | `200` | randomized cases per test. |

CI (`.github/workflows/ci.yml`) runs the gate on every push to `main`/`develop`
across gcc/gfortran **-12 and -15** and the seed rotation `1 7 42 1234
20240617`. Clear a fix locally the same way:

```bash
for s in 1 7 42 1234 20240617; do BLAS_FUZZ_SEED=$s ctest --test-dir build -L fuzz -R <routine> || break; done
```

## Performance

A change to a hot path is not done until it is re-timed. Performance has its own
doc — [benchmark.md](benchmark.md) — covering the dual-link harness, the
pass/fail bars, and how to run it, with the full runbook and rationale in
`bench/dual/`. The everyday cycle after a kernel change:

```bash
bench/dual/update_routine.sh e etbsv     # re-time + refresh reports/dual_scoreboard.md
```
