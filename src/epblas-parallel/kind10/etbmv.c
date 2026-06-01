/*
 * etbmv — kind10 (long double) triangular band matrix-vector.
 *   x := A*x or A^T*x, A triangular band with K+1 diagonals.
 *
 * The serial sweep is this overlay's faster reference and is defined FIRST so
 * its hot-loop placement is unperturbed by the threaded scaffolding (function
 * order shifts alignment). x := op(A)*x is a pure matvec (no solve dependency),
 * so although the in-place column sweep serializes it, every OUTPUT element is
 * independent. The threaded path (large N) exploits that with a ROW-PARTITIONED
 * GATHER: thread t owns a disjoint output-row range [lo,hi) and computes each
 * y[i] as a band dot accumulated in a register-resident x87 scalar, reading x
 * globally but never writing it; one barrier; then each thread copies its own
 * range back to x. No per-thread buffer, no zero-fill, no O(n*nthreads)
 * reduction (which is what OpenBLAS tbmv_thread pays AND what an earlier private-
 * slot port here paid) -- the only serial cost is one malloc(n), so speedup
 * climbs monotonically toward nthreads instead of capping. The gather also runs
 * ~1.7x faster per element than the in-place scatter (accumulator stays off
 * memory). The orchestration is a noinline helper so its bookkeeping does not
 * crowd the serial sweep's x87 allocator.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Thread-entry thresholds = the measured break-even where par4 < par1 (forking
 * actually beats our own serial sweep). Trans needs ~2x the size: its gather
 * kernel is ~half the work per row, so the fixed omp-parallel fork/join cost
 * amortizes at larger N. Below these, the serial sweep is faster than any thread
 * split — and faster than ob's threaded path too — so we never fork into a loss. */
#define ETBMV_OMP_MIN_N 512   /* NoTrans/ConjNoTrans: break-even ~N=350 */
#define ETBMV_OMP_MIN_T 1024  /* Trans: break-even ~N=800 */
#define ETBMV_MAX_CPUS 256

#ifdef _OPENMP
static int etbmv_omp(int upper, int trans, int nounit, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx);
#endif

void etbmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) short-circuit
     * before the noinline call's argument marshalling (outlining tax). */
    const int omp_min = (TR == 'T') ? ETBMV_OMP_MIN_T : ETBMV_OMP_MIN_N;
    if (N >= omp_min && blas_omp_max_threads() > 1
        && etbmv_omp(UPLO == 'U', TR == 'T', nounit, N, K, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) x[i] += tmp * A_(L + i, j);
                        if (nounit) x[j] *= A_(K, j);
                    }
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) x[i] += tmp * A_(i - j, j);
                        if (nounit) x[j] *= A_(0, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp *= A_(K, j);
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= A_(0, j);
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) tmp += A_(i - j, j) * x[i];
                    x[j] = tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) {
                            x[ix] += tmp * A_(L + i, j);
                            ix += incx;
                        }
                        if (nounit) x[jx] *= A_(K, j);
                    }
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) {
                            x[ix] += tmp * A_(i - j, j);
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= A_(0, j);
                    }
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    int ix = kx;
                    const int L = K - j;
                    if (nounit) tmp *= A_(K, j);
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        tmp += A_(L + i, j) * x[ix];
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    int ix = kx;
                    if (nounit) tmp *= A_(0, j);
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
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
 * resident scalar (x87 stack), then a single store y[i]=s. This is the lever:
 * the in-place serial sweep SCATTERS (read-modify-write to memory per element),
 * but op(A)*x has independent output rows, so gathering each row as a dot keeps
 * the accumulator off memory and is ~1.7x faster per element than the scatter
 * AND lets us partition output rows disjointly — no per-thread buffer, no
 * zero-fill, no reduction. Branch hoisted out of the i-loop; lda-1 (NoTrans
 * anti-diagonal stride) vs contiguous (Trans). */
static void tbmv_rowgather(int upper, int trans, int nounit,
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
__attribute__((noinline)) static int etbmv_omp(
    int upper, int trans, int nounit, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx)
{
    ptrdiff_t omp_min = trans ? ETBMV_OMP_MIN_T : ETBMV_OMP_MIN_N;
    if (n < omp_min || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
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
        int tid = omp_get_thread_num();
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

#undef A_
