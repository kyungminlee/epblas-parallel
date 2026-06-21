/*
 * mf_omp.h — shared OpenMP work-partition helpers for the multifloats BLAS.
 *
 * tri_area_bounds(): split [0,n) into ~nthreads contiguous slices carrying equal
 * TRIANGULAR work. The per-index work grows (or shrinks) linearly, so cumulative
 * work is quadratic; equal-area widths solve that quadratic. Used by every
 * threaded triangular/symmetric-packed matvec whose per-thread cost scales with
 * the triangle, where an equal-WIDTH split would skew load ~2x.
 *
 *   heavy_high=true   work per index grows with the index (e.g. UPPER symv/trmv:
 *                     column j touches rows 0..j) → thin slices at the high end,
 *                     boundaries b_k = sqrt(k * n*n/nthreads).
 *   heavy_high=false  work shrinks (LOWER) → thin slices at the low end.
 *
 * The two genuine consumers — mspmv (packed symv, column partition, read forward)
 * and wtrmv (dense trmv NoTrans, row partition; UPPER reads the slices reversed
 * so the heavy top rows land on the thin top slice) — produce the SAME equal-area
 * boundary set; they only differ in fill/read direction and tuning constants
 * (mask/min_width), which stay per-consumer parameters here. (msbmv is NOT a
 * client: symmetric-BAND work is uniform per column, so it splits equal-width.)
 *
 * Runs once per call (setup, never in a SIMD loop) → perf-neutral. Fills
 * range[0..return value] ascending; returns the slice count (<= max_cpus).
 *
 * tri_row_window()/band_row_window(): the matching per-thread fold window. After
 * each thread scatters its column slice into a private slot, the bounded
 * reduction walks only the rows that slot can have touched. Every consumer folds
 * with the SAME idiom — accumulate each slot's window straight onto the output
 * (y[k] += alpha*slot[k], or x[k]=Σslot[k] for the tpmv/trmv cases whose output
 * aliases the input) — so the window math is the only thing that varies, and it
 * lives here once instead of being re-derived (and risking an off-by-one) per
 * file.
 */
#pragma once

#include <cmath>
#include <cstddef>

namespace mf_omp {

static inline int tri_area_bounds(std::ptrdiff_t n, int nthreads,
                                  int mask, int min_width, bool heavy_high,
                                  int max_cpus, std::ptrdiff_t *range)
{
    int num_cpu = 0;
    const double dnum = (double)n * (double)n / (double)nthreads;
    range[0] = 0;
    std::ptrdiff_t i = 0;
    while (i < n) {
        std::ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            if (heavy_high) {
                /* grow root: width = sqrt(i^2 + dnum) - i (widths shrink as i
                 * rises, so the heavy high end gets the thin slices). */
                double di = (double)i;
                width = (std::ptrdiff_t)(std::sqrt(di * di + dnum) - di);
            } else {
                /* shrink root: width = (n-i) - sqrt((n-i)^2 - dnum). */
                double di = (double)(n - i);
                double rad = di * di - dnum;
                if (rad > 0.0) width = (std::ptrdiff_t)(-std::sqrt(rad) + di);
                else           width = n - i;
            }
            width = (width + mask) & ~(std::ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i += width;
        if (num_cpu >= max_cpus) break;
    }
    return num_cpu;
}

/* band_bounds(): split [0,n) into ~nthreads contiguous EQUAL-WIDTH slices. For
 * BANDED symmetric/Hermitian matvec the per-column work is uniform (~2K band
 * entries, independent of the column index), so an equal split balances load — a
 * triangular sqrt split (tri_area_bounds) would skew it. Widths are rounded up to
 * (mask+1) and floored at min_width so per-thread slot writes stay off shared
 * cache lines. Fills range[0..return]; returns the slice count (<= max_cpus). */
static inline int band_bounds(std::ptrdiff_t n, int nthreads, int mask,
                              int min_width, int max_cpus, std::ptrdiff_t *range)
{
    int num_cpu = 0;
    const std::ptrdiff_t base = (n + nthreads - 1) / nthreads;
    range[0] = 0;
    std::ptrdiff_t i = 0;
    while (i < n) {
        std::ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            width = (base + mask) & ~(std::ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i += width;
        if (num_cpu >= max_cpus) break;
    }
    return num_cpu;
}

/* tri_row_window(): the populated ROW window of thread t's private slot for a
 * COLUMN-partitioned triangular/symmetric/Hermitian matvec. Thread t owns
 * columns [range[t],range[t+1]); for UPPER storage column j scatters into rows
 * 0..j, so the slot is touched over [0,range[t+1]); for LOWER, rows j..n-1 ->
 * [range[t],n). The shared bounded-reduction fold walks exactly this window, so
 * the off-by-one logic lives here once instead of in every consumer. */
static inline void tri_row_window(int t, bool upper, const std::ptrdiff_t *range,
                                  std::ptrdiff_t n,
                                  std::ptrdiff_t &from, std::ptrdiff_t &to)
{
    from = upper ? 0 : range[t];
    to   = upper ? range[t + 1] : n;
}

/* band_row_window(): the populated ROW window of thread t's slot for a BAND
 * (equal-width) matvec with half-bandwidth k. Thread t's columns
 * [range[t],range[t+1]) touch rows within k of that span: UPPER
 * [range[t]-k,range[t+1]); LOWER [range[t],range[t+1]+k), clamped to [0,n).
 * Adjacent windows overlap by k, but each column's contribution lives in exactly
 * one slot, so summing the overlapping windows is correct. */
static inline void band_row_window(int t, bool upper, const std::ptrdiff_t *range,
                                   std::ptrdiff_t n, std::ptrdiff_t k,
                                   std::ptrdiff_t &from, std::ptrdiff_t &to)
{
    if (upper) { from = range[t] - k; if (from < 0) from = 0; to = range[t + 1]; }
    else       { from = range[t]; to = range[t + 1] + k; if (to > n) to = n; }
}

/* even_slice(): the flat EQUAL-COUNT partition of [0,n) used by every threaded L1
 * / reduction sweep (copy/scal/axpy/rot/asum/dot/nrm2/argmax). Thread tid of nth
 * gets [lo,hi) = [n*tid/nth, n*(tid+1)/nth); the long-long product avoids int
 * overflow before the divide. Byte-identical math previously re-derived in ~20
 * files — centralized here so the slice bounds (and their overflow guard) live
 * once. */
static inline void even_slice(int n, int tid, int nth, int &lo, int &hi)
{
    lo = (int)((long long)n * tid / nth);
    hi = (int)((long long)n * (tid + 1) / nth);
}

}  // namespace mf_omp
