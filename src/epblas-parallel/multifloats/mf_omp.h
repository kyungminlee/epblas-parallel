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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common/blas_omp.h"  /* the single partition primitive blas_part_bound */

namespace mf_omp {

static inline std::ptrdiff_t tri_area_bounds(std::ptrdiff_t n, std::ptrdiff_t nthreads,
                                  std::ptrdiff_t mask, std::ptrdiff_t min_width, bool heavy_high,
                                  std::ptrdiff_t max_cpus, std::ptrdiff_t *range)
{
    std::ptrdiff_t num_cpu = 0;
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
static inline std::ptrdiff_t band_bounds(std::ptrdiff_t n, std::ptrdiff_t nthreads, std::ptrdiff_t mask,
                              std::ptrdiff_t min_width, std::ptrdiff_t max_cpus, std::ptrdiff_t *range)
{
    std::ptrdiff_t num_cpu = 0;
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
static inline void tri_row_window(std::ptrdiff_t t, bool upper, const std::ptrdiff_t *range,
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
static inline void band_row_window(std::ptrdiff_t t, bool upper, const std::ptrdiff_t *range,
                                   std::ptrdiff_t n, std::ptrdiff_t k,
                                   std::ptrdiff_t &from, std::ptrdiff_t &to)
{
    if (upper) { from = range[t] - k; if (from < 0) from = 0; to = range[t + 1]; }
    else       { from = range[t]; to = range[t + 1] + k; if (to > n) to = n; }
}

/* even_slice(): the flat balanced partition of [0,n) used by every threaded L1
 * / reduction sweep (copy/scal/axpy/rot/asum/dot/nrm2/argmax). Thread tid of nth
 * gets [lo,hi) via the single front-loaded-remainder primitive blas_part_bound
 * (common/blas_omp.h) — overflow-safe in pure 64-bit, no __int128/soft-division.
 * Byte-identical math previously re-derived in ~20 files; centralized here so the
 * slice bounds route through the one partition primitive. */
static inline void even_slice(std::ptrdiff_t n, std::ptrdiff_t tid, std::ptrdiff_t nth, std::ptrdiff_t &lo, std::ptrdiff_t &hi)
{
    lo = blas_part_bound(n, tid, nth);
    hi = blas_part_bound(n, tid + 1, nth);
}

#ifdef _OPENMP
/* ---- Shared L1 partial[]-reduce wrappers (asum/dot/nrm2/argmax) ----------
 *
 * L1_MAX_CPUS caps the requested team size AND sizes the per-call partial[]
 * (resp. idx[]/val[]) stack arrays. Deliberately 64, NOT the 256 used by the
 * L2 matvec partitions (the various *_MAX_CPUS = 256 range[] caps): L1
 * reductions are bandwidth-bound and gain nothing past a few tens of threads,
 * and the smaller cap keeps the stack arrays tiny. The 64-vs-256 split is
 * intentional — do not unify. */
constexpr std::ptrdiff_t L1_MAX_CPUS = 64;

/* l1_team(): the requested team size for the L1 wrappers — global max threads
 * clamped to L1_MAX_CPUS. The ACTUAL team the runtime delivers
 * (omp_get_num_threads()) may be smaller (omp_dynamic, resource pressure),
 * which is why the wrappers below pre-initialize every requested slot. */
static inline std::ptrdiff_t l1_team()
{
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > L1_MAX_CPUS) nthreads = L1_MAX_CPUS;
    return nthreads;
}

/* partial_reduce(): the shared threaded sum-reduction. Thread tid runs
 * slice(lo,hi) over its even_slice of [0,n) and writes partial[tid]; the merge
 * folds partial[] in ascending-tid order — the same per-thread accumulation
 * and cross-thread merge order the per-file copies used, so results are
 * bit-identical on a full team. Every REQUESTED slot is pre-initialized to
 * `zero` BEFORE the parallel region: the merge walks the requested nthreads,
 * but only the ACTUAL team writes its slots, so without the pre-fill a
 * short-delivered team left uninitialized stack reads in the merge. Folding
 * the neutral `zero` for an unwritten slot is exact (DD add of a true zero is
 * the identity). */
template <typename T, typename Slice, typename Fold>
static inline T partial_reduce(std::ptrdiff_t n, const T &zero, Slice &&slice, Fold &&fold)
{
    const std::ptrdiff_t nthreads = l1_team();
    T partial[L1_MAX_CPUS];
    for (std::ptrdiff_t t = 0; t < nthreads; ++t) partial[t] = zero;
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; even_slice(n, tid, nth, lo, hi);
        if (lo < hi) partial[tid] = slice(lo, hi);
    }
    T s = zero;
    for (std::ptrdiff_t t = 0; t < nthreads; ++t) s = fold(s, partial[t]);
    return s;
}

/* partial_argmax(): the shared threaded 1-based argmax. scan(lo,hi,bv) must
 * return the 1-BASED global index of its slice winner and store the winning
 * magnitude in bv. idx slots are pre-initialized to the sentinel 0 ("no
 * entry") and val slots to `vzero` BEFORE the parallel region, so slots a
 * short-delivered team never writes are skipped by the merge and can never
 * win (or tie) against a real entry. Ascending-tid merge with a STRICT
 * greater test keeps the LOWEST global index on ties — bit-identical to the
 * serial left-to-right scan. */
template <typename V, typename Scan, typename Greater>
static inline std::ptrdiff_t partial_argmax(std::ptrdiff_t n, const V &vzero,
                                            Scan &&scan, Greater &&greater)
{
    const std::ptrdiff_t nthreads = l1_team();
    std::ptrdiff_t idx[L1_MAX_CPUS];
    V val[L1_MAX_CPUS];
    for (std::ptrdiff_t t = 0; t < nthreads; ++t) { idx[t] = 0; val[t] = vzero; }
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; even_slice(n, tid, nth, lo, hi);
        if (lo < hi) {
            V bv = vzero;
            idx[tid] = scan(lo, hi, bv);
            val[tid] = bv;
        }
    }
    std::ptrdiff_t best = 0;
    V bestv = vzero;
    for (std::ptrdiff_t t = 0; t < nthreads; ++t) {
        if (idx[t] == 0) continue;
        if (best == 0 || greater(val[t], bestv)) { best = idx[t]; bestv = val[t]; }
    }
    return best;
}
#endif  /* _OPENMP */

}  // namespace mf_omp
