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
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef long double T;


#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Thread-entry thresholds = the measured break-even where par4 < par1 (forking
 * actually beats our own serial path). Both serial paths are register-resident
 * (NoTrans in-place gather, Trans dot), so threading only pays once N is large
 * enough to amortize the threaded path's malloc(n) + copy-back + fork/join.
 * Below these the serial path wins — and beats ob's threaded path too — so we
 * never fork into a loss. NoTrans's fast in-place-gather serial pushes its
 * break-even up near Trans's (it was 512 back when serial NoTrans was the slower
 * column scatter). Calibrated at K=16. */
#define ETBMV_OMP_MIN_N 768   /* NoTrans/ConjNoTrans: break-even ~N=700 */
#define ETBMV_OMP_MIN_T 1024  /* Trans: break-even ~N=800 */
#define ETBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t etbmv_omp(bool upper, bool trans, bool nounit, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx);
#endif

static void etbmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    char TR = blas_up(trans);
    if (TR == 'C') TR = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) short-circuit
     * before the noinline call's argument marshalling (outlining tax). */
    const ptrdiff_t omp_min = (TR == 'T') ? ETBMV_OMP_MIN_T : ETBMV_OMP_MIN_N;
    if (n >= omp_min && blas_omp_max_threads() > 1
        && etbmv_omp(UPLO == 'U', TR == 'T', nounit, n, k, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            /* In-place row-gather (NOT the old column scatter): each output x[i]
             * is a band dot accumulated in a register-resident x87 scalar and
             * stored once -- no read-modify-write to memory per element. Safe
             * with no buffer because upper reads indices >= i and lower reads
             * indices <= i, so upper ASCENDING / lower DESCENDING never clobbers
             * an unread input. base[k +- d*s1] walks the lda-1 anti-diagonal. */
            const ptrdiff_t s1 = lda - 1;
            if (UPLO == 'U') {
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const T *base = &A_(0, i);
                    const ptrdiff_t len = (n - 1 - i < k) ? (n - 1 - i) : k;
                    T s = nounit ? base[k] * x[i] : x[i];
                    for (ptrdiff_t d = 1; d <= len; ++d) s += base[k + (ptrdiff_t)d * s1] * x[i + d];
                    x[i] = s;
                }
            } else {
                for (ptrdiff_t i = n - 1; i >= 0; --i) {
                    const T *base = &A_(0, i);
                    const ptrdiff_t len = (i < k) ? i : k;
                    T s = nounit ? base[0] * x[i] : x[i];
                    for (ptrdiff_t d = 1; d <= len; ++d) s += base[-(ptrdiff_t)d * s1] * x[i - d];
                    x[i] = s;
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const ptrdiff_t L = k - j;
                    if (nounit) tmp *= A_(k, j);
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= A_(0, j);
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = j + 1; i < i_hi; ++i) tmp += A_(i - j, j) * x[i];
                    x[j] = tmp;
                }
            }
        }
    } else {
        if (TR == 'N') {
            /* Strided in-place row-gather (same as the incx==1 path; logical
             * index i lives at x[off0 + i*incx], off0 placing logical 0 for
             * incx<0). Register-resident accumulator; upper ASCENDING / lower
             * DESCENDING keeps the in-place write safe. */
            const ptrdiff_t off0 = (incx < 0) ? -(ptrdiff_t)(n - 1) * incx : 0;
            const ptrdiff_t s1 = lda - 1;
            if (UPLO == 'U') {
                for (ptrdiff_t i = 0; i < n; ++i) {
                    const T *base = &A_(0, i);
                    const ptrdiff_t len = (n - 1 - i < k) ? (n - 1 - i) : k;
                    const ptrdiff_t ii = off0 + (ptrdiff_t)i * incx;
                    T s = nounit ? base[k] * x[ii] : x[ii];
                    ptrdiff_t ix = ii + incx;
                    for (ptrdiff_t d = 1; d <= len; ++d) { s += base[k + (ptrdiff_t)d * s1] * x[ix]; ix += incx; }
                    x[ii] = s;
                }
            } else {
                for (ptrdiff_t i = n - 1; i >= 0; --i) {
                    const T *base = &A_(0, i);
                    const ptrdiff_t len = (i < k) ? i : k;
                    const ptrdiff_t ii = off0 + (ptrdiff_t)i * incx;
                    T s = nounit ? base[0] * x[ii] : x[ii];
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
                    T tmp = x[jx];
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
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    ptrdiff_t ix = kx;
                    if (nounit) tmp *= A_(0, j);
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                        tmp += A_(i - j, j) * x[ix];
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
                           const T *restrict a, ptrdiff_t lda,
                           const T *restrict x, T *restrict y)
{
    const ptrdiff_t s1 = lda - 1;               /* NoTrans diagonal-walk stride */
    if (!trans) {
        if (upper) {                             /* y[i] = A(i,i)x[i] + Σ A(i,i+d)x[i+d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                T s = nounit ? base[k] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[k + d * s1] * x[i + d];
                y[i] = s;
            }
        } else {                                 /* y[i] = A(i,i)x[i] + Σ A(i,i-d)x[i-d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (i < k) ? i : k;
                T s = nounit ? base[0] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[-d * s1] * x[i - d];
                y[i] = s;
            }
        }
    } else {
        if (upper) {                             /* y[i] = A(i,i)x[i] + Σ A(i-d,i)x[i-d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (i < k) ? i : k;
                T s = nounit ? base[k] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[k - d] * x[i - d];
                y[i] = s;
            }
        } else {                                 /* y[i] = A(i,i)x[i] + Σ A(i+d,i)x[i+d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                T s = nounit ? base[0] * x[i] : x[i];
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
    const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx)
{
    ptrdiff_t omp_min = trans ? ETBMV_OMP_MIN_T : ETBMV_OMP_MIN_N;
    if (n < omp_min || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > ETBMV_MAX_CPUS) nthreads = ETBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;

    /* Gather strided input to contiguous; y is the disjoint output scratch
     * (written once per element, so no zero-fill). */
    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }
    T *y = (T *)malloc((size_t)n * sizeof(T));
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

EPBLAS_FACADE_TBMV(etbmv, T)

#undef A_
