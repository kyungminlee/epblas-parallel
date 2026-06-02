/*
 * ytbmv — kind10 complex triangular band matrix-vector.
 *   x := op(A)*x, op = none / transpose / conjugate-transpose.
 *
 * Complex twin of etbmv. x := op(A)*x is a pure matvec (no solve dependency):
 * every OUTPUT element is an independent band dot. Both the serial reference and
 * the threaded path therefore GATHER -- each x[i] is accumulated in a register-
 * resident x87 scalar and stored once, never the column SCATTER (read-modify-
 * write to memory per element) that reference tbmv uses for NoTrans. Keeping the
 * accumulator off memory runs ~2x faster per element (project_x87_accumulator_spill).
 *
 * Serial NoTrans gathers IN PLACE with no buffer: op(A)*x reads only indices
 * >= i (upper) or <= i (lower), so sweeping upper ASCENDING / lower DESCENDING
 * never clobbers an unread input. Serial Trans/ConjTrans is already a register
 * dot (conjugating the band/diagonal entries when ConjTrans).
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
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef _Complex long double T;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Thread-entry threshold = the measured break-even where par4 < par1 (forking
 * actually beats our own serial path). Both serial paths are register-resident
 * (NoTrans in-place gather, Trans/ConjTrans dot), so threading only pays once N
 * is large enough to amortize the threaded path's malloc(n) + copy-back +
 * fork/join. Complex arithmetic is heavy enough per element that ALL three ops
 * (NoTrans/Trans/ConjTrans) share one break-even (~N=290) -- no per-op split
 * like etbmv's real case (768 NoTrans vs 1024 Trans). The fast in-place-gather
 * serial pushes the break-even up; measured against the real-codegen serial,
 * N=256 is still a marginal loss for unit-diag/Trans and N=320 is the first N
 * where every shape clearly wins (~0.85-0.89). Calibrated at K=16. */
#ifndef YTBMV_OMP_MIN
#define YTBMV_OMP_MIN 320   /* first N where par4 < par1 for all ops */
#endif
#define YTBMV_MAX_CPUS 256

#ifdef _OPENMP
static int ytbmv_omp(int upper, int trans, int conj, int nounit,
                     ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx);
#endif

void ytbmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) short-circuit
     * before the noinline call's argument marshalling (outlining tax). */
    if (N >= YTBMV_OMP_MIN && blas_omp_max_threads() > 1
        && ytbmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, K, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            /* In-place row-gather (NOT the old column scatter): each output x[i]
             * is a band dot accumulated in a register-resident x87 scalar and
             * stored once -- no read-modify-write to memory per element. Safe
             * with no buffer because upper reads indices >= i and lower reads
             * indices <= i, so upper ASCENDING / lower DESCENDING never clobbers
             * an unread input. base[k +- d*s1] walks the lda-1 anti-diagonal.
             * NoTrans never conjugates. */
            const ptrdiff_t s1 = lda - 1;
            if (UPLO == 'U') {
                for (int i = 0; i < N; ++i) {
                    const T *base = &A_(0, i);
                    const int len = (N - 1 - i < K) ? (N - 1 - i) : K;
                    T s = nounit ? base[K] * x[i] : x[i];
                    for (int d = 1; d <= len; ++d) s += base[K + (ptrdiff_t)d * s1] * x[i + d];
                    x[i] = s;
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    const T *base = &A_(0, i);
                    const int len = (i < K) ? i : K;
                    T s = nounit ? base[0] * x[i] : x[i];
                    for (int d = 1; d <= len; ++d) s += base[-(ptrdiff_t)d * s1] * x[i - d];
                    x[i] = s;
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp *= (noconj ? A_(K, j) : cconj(A_(K, j)));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    if (noconj) for (int i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
                    else        for (int i = j - 1; i >= i_lo; --i) tmp += cconj(A_(L + i, j)) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= (noconj ? A_(0, j) : cconj(A_(0, j)));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    if (noconj) for (int i = j + 1; i < i_hi; ++i) tmp += A_(i - j, j) * x[i];
                    else        for (int i = j + 1; i < i_hi; ++i) tmp += cconj(A_(i - j, j)) * x[i];
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
            const ptrdiff_t off0 = (incx < 0) ? -(ptrdiff_t)(N - 1) * incx : 0;
            const ptrdiff_t s1 = lda - 1;
            if (UPLO == 'U') {
                for (int i = 0; i < N; ++i) {
                    const T *base = &A_(0, i);
                    const int len = (N - 1 - i < K) ? (N - 1 - i) : K;
                    const ptrdiff_t ii = off0 + (ptrdiff_t)i * incx;
                    T s = nounit ? base[K] * x[ii] : x[ii];
                    ptrdiff_t ix = ii + incx;
                    for (int d = 1; d <= len; ++d) { s += base[K + (ptrdiff_t)d * s1] * x[ix]; ix += incx; }
                    x[ii] = s;
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    const T *base = &A_(0, i);
                    const int len = (i < K) ? i : K;
                    const ptrdiff_t ii = off0 + (ptrdiff_t)i * incx;
                    T s = nounit ? base[0] * x[ii] : x[ii];
                    ptrdiff_t ix = ii - incx;
                    for (int d = 1; d <= len; ++d) { s += base[-(ptrdiff_t)d * s1] * x[ix]; ix -= incx; }
                    x[ii] = s;
                }
            }
        } else {
            int kx = (incx < 0) ? -(N - 1) * incx : 0;
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    int ix = kx;
                    const int L = K - j;
                    if (nounit) tmp *= (noconj ? A_(K, j) : cconj(A_(K, j)));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        const T aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
                        tmp += aij * x[ix];
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
                    if (nounit) tmp *= (noconj ? A_(0, j) : cconj(A_(0, j)));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
                        const T aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
                        tmp += aij * x[ix];
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
 * serial paths use, but writing a disjoint scratch buffer rather than in place,
 * so output rows partition across threads with no cross-thread write dependence
 * — no per-thread buffer beyond the shared y, no zero-fill, no reduction.
 * Branch hoisted out of the i-loop; lda-1 (NoTrans anti-diagonal stride) vs
 * contiguous (Trans). ConjTrans conjugates the band and diagonal entries. */
static void tbmv_rowgather(int upper, int trans, int conj, int nounit,
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
                T s = nounit ? (conj ? cconj(base[k]) : base[k]) * x[i] : x[i];
                if (!conj) for (ptrdiff_t d = 1; d <= len; ++d) s += base[k - d] * x[i - d];
                else       for (ptrdiff_t d = 1; d <= len; ++d) s += cconj(base[k - d]) * x[i - d];
                y[i] = s;
            }
        } else {                                 /* y[i] = A(i,i)x[i] + Σ A(i+d,i)x[i+d] */
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                T s = nounit ? (conj ? cconj(base[0]) : base[0]) * x[i] : x[i];
                if (!conj) for (ptrdiff_t d = 1; d <= len; ++d) s += base[d] * x[i + d];
                else       for (ptrdiff_t d = 1; d <= len; ++d) s += cconj(base[d]) * x[i + d];
                y[i] = s;
            }
        }
    }
}

/* Threaded triangular band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): it gathers y[lo,hi) reading x, a barrier, then copies its own range
 * back to x. No cross-thread data dependence beyond the single read/write
 * barrier — near-linear scaling. Returns 1 if handled, 0 to fall back. */
__attribute__((noinline)) static int ytbmv_omp(
    int upper, int trans, int conj, int nounit, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx)
{
    if (n < YTBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > YTBMV_MAX_CPUS) nthreads = YTBMV_MAX_CPUS;

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
        tbmv_rowgather(upper, trans, conj, nounit, n, k, lo, hi, a, lda, xptr, y);
        #pragma omp barrier              /* all reads of x done before any write-back */
        if (incx == 1) for (ptrdiff_t i = lo; i < hi; ++i) x[i] = y[i];
        else           for (ptrdiff_t i = lo; i < hi; ++i) x[i * incx] = y[i];
    }

    free(y); free(xbuf);
    return 1;
}
#endif /* _OPENMP */

#undef A_
