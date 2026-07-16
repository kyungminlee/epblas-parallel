# egemm 5-way interleaved (authoritative)

Warmed interleaved 5-way table, min-of-7 wall ns/call, all five ways alternated
within each rep so thermal/drift cancels. taskset-pinned, `OMP_WAIT_POLICY=passive`.
Square M=N=K, `beta=0`, `alpha=0.9`. Ways: par1 par4 ob1 ob4 mig (mig = migrated
serial reference, from the par archive). **Wall ns/call; ratios SMALLER = par
faster.** Harness `/tmp/l2d/cmp5_egemm.sh`, driver `/tmp/l2d/d5_egemm.c`.

Two fixes landed (HEAD): (1) **TN routed through the blocked path** — the
`ta='T',tb='N'` `fast_col` no-pack dot is gated to skinny K only; (2) **threaded
MC cap** — the top-level entry caps MC so small-M problems split across the team.

```
key     N |        par1        par4         ob1         ob4         mig |   p1/o1   p4/o4   p4/p1  p1/mig
NN    128 |     1944106      570098     1915462      697097     3095998 |   1.015   0.818   0.293   0.628
NN    256 |    14963150     4272597    14951242     4099894    24305799 |   1.001   1.042   0.286   0.616
NN    512 |   119260602    30894922   119113138    30477631   194242125 |   1.001   1.014   0.259   0.614
NN   1024 |   949770247   241743646   954107924   242084604  1714251170 |   0.995   0.999   0.255   0.554

TN    128 |     1950412      574957     1914743      707104     2714709 |   1.019   0.813   0.295   0.718
TN    256 |    14939875     3996215    14994246     3990629    21799062 |   0.996   1.001   0.267   0.685
TN    512 |   118629568    30417822   118842707    30569538   174889710 |   0.998   0.995   0.256   0.678
TN   1024 |   948591880   242767227   953196872   241762820  1712018812 |   0.995   1.004   0.256   0.554

NT    128 |     1937110      538598     1921317      691221     3107513 |   1.008   0.779   0.278   0.623
NT    256 |    14995056     4016992    14973236     4001955    24630598 |   1.001   1.004   0.268   0.609
NT    512 |   118758012    30661642   118879450    30391099   197157471 |   0.999   1.009   0.258   0.602
NT   1024 |   950280246   244090774   954452610   241581144  1733170906 |   0.996   1.010   0.257   0.548
```

After both fixes egemm is at parity-or-better with the OpenBLAS overlay across
NN/NT/TN at every size and both OMP levels: serial `p1/o1` 0.995–1.019, OMP=4
`p4/o4` 0.78–1.04 (par4 *beats* ob4 at N=128). par threads ~3.4–3.9× at N≥256
(`p4/p1` 0.255–0.293) and is ~1.8× faster than the migrated reference serially
(`p1/mig` 0.55–0.72).

## Fix 1 — TN: route the bulk through the blocked path (commit 6c9fba9)

The `ta='T',tb='N'` fast path (`egemm_fast_col`) runs the stride-1 dot with a
SINGLE fp80 accumulator, so the ~3-cyc x87 `fadd` latency serializes the
reduction — ~5.1 cyc/MAC, identical per-FLOP cost to the naive migrated
reference (the old `p1/mig` ≈ 1.00 confirmed it *was* the naive ref). The
blocked packed path keeps four independent MR×NR accumulator chains that hide
that latency (~2.9 cyc/MAC) and threads over M just as well, so it is faster for
any non-trivial K despite the pack. `egemm_pack_A` already handles `ta='T'`, so
TN falls through cleanly. `egemm_tn_use_fast(M,N,K)` keeps `fast_col` only where
packing isn't amortized (K≤64 or a tiny M·N).

The HPC consult corrected my initial "TN is memory-bound" guess: the A re-stream
at N=1024 is only ~10 GB/s effective (this box does ~50 GB/s) — not a DRAM wall.
It was purely accumulator-ILP.

TN before → after: serial `p1/o1` 1.42–1.79 → 0.995–1.020; OMP=4 `p4/o4`
1.39–1.92 → 0.995–1.015; `p1/mig` ~1.00 → 0.55–0.72 (fast_col bypassed). TN now
tracks NN/NT exactly.

## Fix 2 — threaded MC cap for small M (commit 3fb99ce)

The top-level `egemm_` entry partitions the ic loop across the team via
`omp for`, so the ic-block count `ceil(M/MC)` must be ≥ the team size or threads
idle. `egemm_choose_blocks`' adaptive-MC growth (grows MC toward L2 when K≤KC)
collapsed N=K=128 to a SINGLE ic-block (MC grew to ≈M) → par4 ≈ par1, zero
threading. The threaded entry now caps MC so M splits into ≥ nthr blocks. The cap
is local to this entry — the shared `egemm_choose_blocks` policy is untouched, so
the serial path and the L3 routines (esyrk, etrmm, esymm, esyr2k, etrsm) that
partition other axes are unaffected. Only shrinks MC, multiple of MR → results
unchanged.

N=128 OMP=4 `p4/o4` before → after: NN 2.96 → 0.818, NT 2.97 → 0.779,
TN 2.97 → 0.813 (par4 now beats ob4). `p4/p1` 1.04 → ~0.29 (~3.4× threading).
N≥256 unchanged (M/nthr already ≥ default MC).

## What did NOT change (gap #3, serial NN ~3–5% in stale measurements)

My pre-fix baseline was measured against a **stale May-28 `.a`** (the live target
is `libeblas_parallel-gfortran-16.a`). On the current archive the NN/NT serial
and OMP=4-N≥256 gaps were already ~parity (other commits since May 28); the only
gap surviving on develop HEAD was TN's `fast_col` (Fix 1) and N=128 OMP threading
(Fix 2). The MR=NR=2 fp80 microkernel is near the x87 ceiling (2.9 cyc/MAC over
4 fadd chains); widening MR/NR would exceed the 8-deep x87 stack and spill — not
pursued (HPC consult agreed).

## Correctness

`fuzz_egemm` passes at OMP 1 and 4. Independent triple-loop reference
(`/tmp/l2d/chk_egemm.c`), max_rel < 1e-15 across: the TN gate crossover
(K=32/63/64/65/256/1024), NN/NT/TT, rectangular M≠N, N=127/128/130, incx/incy
implicit, OMP 1/4. Large-K TN now sums in KC-block partials (blocked) rather than
one running dot (fast_col), so it is no longer bit-identical to the old path but
stays well within tolerance (max_rel ≈ 9e-19 at K=1024).
