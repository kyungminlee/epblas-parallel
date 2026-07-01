/*
 * etbmv — kind10 (long double) triangular band matrix-vector.
 *   x := A*x or A^T*x, A triangular band with K+1 diagonals.
 *
 * x := op(A)*x is a pure matvec (no solve dependency): every OUTPUT element is
 * an independent band dot. Both the serial reference and the threaded path
 * therefore GATHER -- each x[i] is accumulated in a register-resident x87 scalar
 * and stored once, never the column SCATTER (read-modify-write to memory per
 * element) that reference tbmv uses for NoTrans. Keeping the accumulator off
 * memory runs ~2x faster per element (project_x87_accumulator_spill).
 *
 * Serial NoTrans gathers IN PLACE with no buffer: op(A)*x reads only indices
 * >= i (upper) or <= i (lower), so sweeping upper ASCENDING / lower DESCENDING
 * never clobbers an unread input. Serial Trans is already a register dot.
 *
 * The serial reference is defined FIRST so its hot-loop placement is unperturbed
 * by the threaded scaffolding (function order shifts alignment). The threaded
 * path (large N) gives thread t a disjoint output-row range [lo,hi); it gathers
 * into a shared scratch buffer y, one barrier, then copies its own range back to
 * x. No per-thread buffer, no zero-fill, no O(n*nthreads) reduction (what
 * OpenBLAS tbmv_thread pays AND what an earlier private-slot port here paid) --
 * the only serial cost is one malloc(n), so speedup climbs toward nthreads
 * instead of capping. The orchestration is a noinline helper so its bookkeeping
 * does not crowd the serial reference's x87 allocator.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef long double TR;


#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Contiguous (incx==1) Trans band dot: x := A^T*x in place, each output x[j] a
 * register-resident band dot. NOINLINE so its x87 register allocation is
 * isolated from etbmv_core's surrounding blocks — inlined, the Lower-Trans loop
 * lost ~3-6% (LTU/128 par/ob 1.057) to inline-context regalloc pressure
 * (project_x87_accumulator_spill trigger 3); extracted, it comes to parity-or-
 * better. Upper hoists L=k-j and reads the macro (already wins); Lower hoists
 * the column base pointer (the macro re-derives j*lda per element). */
__attribute__((noinline))
static void etbmv_contig_trans(char UPLO, bool nounit, ptrdiff_t n, ptrdiff_t k,
                               const TR *restrict a, ptrdiff_t lda, TR *restrict x)
{
    if (UPLO == 'U') {
        for (ptrdiff_t j = n - 1; j >= 0; --j) {
            TR tmp = x[j];
            const ptrdiff_t L = k - j;
            if (nounit) tmp *= A_(k, j);
            const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
            for (ptrdiff_t i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
            x[j] = tmp;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            TR tmp = x[j];
            const TR *restrict col = &a[(size_t)j * (size_t)lda];
            if (nounit) tmp *= col[0];
            const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
            for (ptrdiff_t i = j + 1; i < i_hi; ++i) tmp += col[i - j] * x[i];
            x[j] = tmp;
        }
    }
}

/* Thread-entry thresholds = the measured break-even where par4 < par1 (forking
 * actually beats our own serial path). Both serial paths are register-resident
 * (NoTrans in-place gather, Trans dot), so threading only pays once N is large
 * enough to amortize the threaded path's malloc(n) + copy-back + fork/join.
 * Below these the serial path wins — and beats ob's threaded path too — so we
 * never fork into a loss. NoTrans's fast in-place-gather serial pushes its
 * break-even up near Trans's (it was 512 back when serial NoTrans was the slower
 * column scatter). Calibrated at K=16. */
#define ETBMV_OMP_MIN_N 768   /* NoTrans/ConjNoTrans: break-even ~N=700 */
#define ETBMV_OMP_MIN_T 1024  /* Trans strided: break-even ~N=800 (2nd xbuf malloc) */
/* Contiguous Trans breaks even lower under OMP=4: the serial fallback there is
 * NOT free -- it runs in a process with a live 4-thread hot team whose 3 idle
 * spinners draw package power and clip core-2 turbo, inflating the serial path
 * ~10% (par1 6915 -> par4 7654 at N=512) even though it never enters a parallel
 * region. Threading at 512 reclaims those cores; the fork/malloc/barrier tax is
 * repaid by running on 4 non-throttled cores instead of 1 throttled one. Only
 * the contiguous path (single malloc(n)) qualifies; strided keeps 1024. K=16. */
#define ETBMV_OMP_MIN_TC 512  /* Trans, incx==1: break-even under OMP=4 hot team */
#define ETBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t etbmv_omp(bool upper, bool trans, bool nounit, ptrdiff_t n, ptrdiff_t k,
                     const TR *restrict a, ptrdiff_t lda, TR *restrict x, ptrdiff_t incx);
#endif

static void etbmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const TR *restrict a, ptrdiff_t lda,
    TR *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) short-circuit
     * before the noinline call's argument marshalling (outlining tax). */
    const ptrdiff_t omp_min = (TRANS == 'T')
        ? (incx == 1 ? ETBMV_OMP_MIN_TC : ETBMV_OMP_MIN_T)
        : ETBMV_OMP_MIN_N;
    if (n >= omp_min && blas_omp_max_threads() > 1
        && etbmv_omp(UPLO == 'U', TRANS == 'T', nounit, n, k, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TRANS == 'N') {
            /* In-place row-gather (NOT the old column scatter): each output x[i]
             * is a band dot accumulated in a register-resident x87 scalar and
             * stored once -- no read-modify-write to memory per element. Safe
             * with no buffer because upper reads indices >= i and lower reads
             * indices <= i, so upper ASCENDING / lower DESCENDING never clobbers
             * an unread input. base[k +- d*s1] walks the lda-1 anti-diagonal. */
            const ptrdiff_t s1 = lda - 1;
            if (UPLO == 'U') {
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const TR *base = &A_(0, i);
                    const ptrdiff_t len = (n - 1 - i < k) ? (n - 1 - i) : k;
                    TR s = nounit ? base[k] * x[i] : x[i];
                    for (ptrdiff_t d = 1; d <= len; ++d) s += base[k + (ptrdiff_t)d * s1] * x[i + d];
                    x[i] = s;
                }
            } else {
                for (ptrdiff_t i = n - 1; i >= 0; --i) {
                    const TR *base = &A_(0, i);
                    const ptrdiff_t len = (i < k) ? i : k;
                    TR s = nounit ? base[0] * x[i] : x[i];
                    for (ptrdiff_t d = 1; d <= len; ++d) s += base[-(ptrdiff_t)d * s1] * x[i - d];
                    x[i] = s;
                }
            }
        } else {
            etbmv_contig_trans(UPLO, nounit, n, k, a, lda, x);
        }
    } else {
        if (TRANS == 'N') {
            /* Strided in-place row-gather (same as the incx==1 path; logical
             * index i lives at x[off0 + i*incx], off0 placing logical 0 for
             * incx<0). Register-resident accumulator; upper ASCENDING / lower
             * DESCENDING keeps the in-place write safe. */
            const ptrdiff_t off0 = (incx < 0) ? -(ptrdiff_t)(n - 1) * incx : 0;
            const ptrdiff_t s1 = lda - 1;
            if (UPLO == 'U') {
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const TR *base = &A_(0, i);
                    const ptrdiff_t len = (n - 1 - i < k) ? (n - 1 - i) : k;
                    const ptrdiff_t ii = off0 + (ptrdiff_t)i * incx;
                    TR s = nounit ? base[k] * x[ii] : x[ii];
                    ptrdiff_t ix = ii + incx;
                    for (ptrdiff_t d = 1; d <= len; ++d) { s += base[k + (ptrdiff_t)d * s1] * x[ix]; ix += incx; }
                    x[ii] = s;
                }
            } else {
                for (ptrdiff_t i = n - 1; i >= 0; --i) {
                    const TR *base = &A_(0, i);
                    const ptrdiff_t len = (i < k) ? i : k;
                    const ptrdiff_t ii = off0 + (ptrdiff_t)i * incx;
                    TR s = nounit ? base[0] * x[ii] : x[ii];
                    ptrdiff_t ix = ii - incx;
                    for (ptrdiff_t d = 1; d <= len; ++d) { s += base[-(ptrdiff_t)d * s1] * x[ix]; ix -= incx; }
                    x[ii] = s;
                }
            }
        } else {
            ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
            if (UPLO == 'U') {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR tmp = x[jx];
                    kx -= incx;
                    ptrdiff_t ix = kx;
                    const ptrdiff_t L = k - j;
                    if (nounit) tmp *= A_(k, j);
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                        tmp += A_(L + i, j) * x[ix];
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                /* Lower strided Trans: hoist the column base pointer once per
                 * column (col = &a[j*lda]) and walk col[i-j] — the A_(i,j) macro
                 * re-derives (size_t)j*lda at every band element, leaving par
                 * ~3% behind the refs on these strided cells (the same fix that
                 * brought etbsv's strided cells to parity). The Upper strided
                 * branch above already wins with the macro form, so it is left
                 * untouched (measured per-shape, do not unify). */
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR tmp = x[jx];
                    kx += incx;
                    ptrdiff_t ix = kx;
                    const TR *restrict col = &a[(size_t)j * (size_t)lda];
                    if (nounit) tmp *= col[0];
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                        tmp += col[i - j] * x[ix];
                        ix += incx;
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned gather kernel: compute y[i] for i in [lo, hi) as a band dot,
 * reading x globally (never written here) and accumulating in a register-
 * resident scalar (x87 stack), then a single store y[i]=s. Same gather the
 * serial NoTrans path uses, but writing a disjoint scratch buffer rather than in
 * place, so output rows partition across threads with no cross-thread write
 * dependence — no per-thread buffer beyond the shared y, no zero-fill, no
 * reduction. Branch hoisted out of the i-loop; lda-1 (NoTrans anti-diagonal
 * stride) vs contiguous (Trans). */
static void tbmv_rowgather(bool upper, bool trans, bool nounit,
                           ptrdiff_t n, ptrdiff_t k, ptrdiff_t lo, ptrdiff_t hi,
                           const TR *restrict a, ptrdiff_t lda,
                           const TR *restrict x, TR *restrict y)
{
    const ptrdiff_t s1 = lda - 1;               /* NoTrans diagonal-walk stride */
    if (!trans) {
        if (upper) {                             /* y[i] = A(i,i)x[i] + Σ A(i,i+d)x[i+d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const TR *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                TR s = nounit ? base[k] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[k + d * s1] * x[i + d];
                y[i] = s;
            }
        } else {                                 /* y[i] = A(i,i)x[i] + Σ A(i,i-d)x[i-d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const TR *base = &A_(0, i);
                ptrdiff_t len = (i < k) ? i : k;
                TR s = nounit ? base[0] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[-d * s1] * x[i - d];
                y[i] = s;
            }
        }
    } else {
        if (upper) {                             /* y[i] = A(i,i)x[i] + Σ A(i-d,i)x[i-d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const TR *base = &A_(0, i);
                ptrdiff_t len = (i < k) ? i : k;
                TR s = nounit ? base[k] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[k - d] * x[i - d];
                y[i] = s;
            }
        } else {                                 /* y[i] = A(i,i)x[i] + Σ A(i+d,i)x[i+d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const TR *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                TR s = nounit ? base[0] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[d] * x[i + d];
                y[i] = s;
            }
        }
    }
}

/* Threaded triangular band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): it gathers y[lo,hi) reading x, a barrier, then copies its own range
 * back to x. No cross-thread data dependence beyond the single read/write
 * barrier — near-linear scaling. Returns 1 if handled, 0 to fall back. */
__attribute__((noinline)) static ptrdiff_t etbmv_omp(
    bool upper, bool trans, bool nounit, ptrdiff_t n, ptrdiff_t k,
    const TR *restrict a, ptrdiff_t lda, TR *restrict x, ptrdiff_t incx)
{
    ptrdiff_t omp_min = trans ? (incx == 1 ? ETBMV_OMP_MIN_TC : ETBMV_OMP_MIN_T)
                              : ETBMV_OMP_MIN_N;
    if (n < omp_min || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > ETBMV_MAX_CPUS) nthreads = ETBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;

    /* Gather strided input to contiguous; y is the disjoint output scratch
     * (written once per element, so no zero-fill). */
    const TR *xptr = x;
    TR *xbuf = NULL;
    if (incx != 1) {
        xbuf = (TR *)malloc((size_t)n * sizeof(TR));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }
    TR *y = (TR *)malloc((size_t)n * sizeof(TR));
    if (!y) { free(xbuf); return 0; }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = (n * (ptrdiff_t)tid) / nthreads;
        ptrdiff_t hi = (n * (ptrdiff_t)(tid + 1)) / nthreads;
        tbmv_rowgather(upper, trans, nounit, n, k, lo, hi, a, lda, xptr, y);
        #pragma omp barrier              /* all reads of x done before any write-back */
        if (incx == 1) for (ptrdiff_t i = lo; i < hi; ++i) x[i] = y[i];
        else           for (ptrdiff_t i = lo; i < hi; ++i) x[i * incx] = y[i];
    }

    free(y); free(xbuf);
    return 1;
}
#endif /* _OPENMP */

EPBLAS_FACADE_TBMV(etbmv, TR)

#undef A_
