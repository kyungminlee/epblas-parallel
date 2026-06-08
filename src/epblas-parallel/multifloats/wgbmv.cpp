/* wgbmv — multifloats complex DD general band matrix-vector multiply.
 *
 * Reference algorithm + OMP over j on T/C-path only.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
#define WGBMV_OMP_MIN 64
#define WGBMV_MAX_CPUS 256
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
const T czero{ rzero, rzero };
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0
        && dd_iszero(x.im);
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* Threaded NoTrans complex-DD band matvec (NoTrans never conjugates). Output rows
 * partition across threads (each y[i] an independent band dot, x and y distinct).
 * ax[j]=alpha*x[j] precomputed once; seeding with y[i] then adding in ascending-j
 * order reproduces the serial scatter's association -> bit-identical for contiguous
 * and strided. NoTrans reads N of x, writes M of y. Returns true if handled. */
static bool wgbmv_n_omp(int M, int N, int KL, int KU, T alpha,
                        const T *a, int lda,
                        const T *x, int incx, T *y, int incy)
{
    if (M < WGBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WGBMV_MAX_CPUS) nthreads = WGBMV_MAX_CPUS;

    T *ax = static_cast<T *>(std::malloc(static_cast<std::size_t>(N) * sizeof(T)));
    if (!ax) return false;
    const int ix0 = (incx < 0) ? -(N - 1) * incx : 0;
    for (int j = 0; j < N; ++j) ax[j] = cmul(alpha, x[ix0 + j * incx]);
    const std::ptrdiff_t iy0 = (incy < 0) ? -static_cast<std::ptrdiff_t>(M - 1) * incy : 0;

    const std::ptrdiff_t s1 = static_cast<std::ptrdiff_t>(lda) - 1;
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        std::ptrdiff_t lo = (static_cast<std::ptrdiff_t>(M) * tid) / nthreads;
        std::ptrdiff_t hi = (static_cast<std::ptrdiff_t>(M) * (tid + 1)) / nthreads;
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            std::ptrdiff_t j_lo = (i - KL > 0) ? (i - KL) : 0;
            std::ptrdiff_t j_hi = (i + KU + 1 < N) ? (i + KU + 1) : N;
            const T *base = a + (KU + i);
            T *yi = &y[iy0 + i * incy];
            T s = *yi;                        /* seed: post-beta y[i] */
            for (std::ptrdiff_t j = j_lo; j < j_hi; ++j) s = cadd(s, cmul(ax[j], base[j * s1]));
            *yi = s;
        }
    }
    std::free(ax);
    return true;
}

/* Threaded Trans/ConjTrans complex-DD band matvec (strided x and/or y). Output
 * columns partition across threads (each y[j]=alpha*Σ_i op(A(i,j))*x[i] disjoint).
 * Strided x gathered to contiguous; noconj selects T (identity) vs C (conjugate).
 * Trans reads M of x, writes N of y. Bit-identical to the serial strided gather. */
static bool wgbmv_t_omp(int M, int N, int KL, int KU, T alpha,
                        const T *a, int lda,
                        const T *x, int incx, T *y, int incy, int noconj)
{
    if (N < WGBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WGBMV_MAX_CPUS) nthreads = WGBMV_MAX_CPUS;

    if (incx < 0) x -= static_cast<std::ptrdiff_t>(M - 1) * incx;
    if (incy < 0) y -= static_cast<std::ptrdiff_t>(N - 1) * incy;

    const T *xptr = x;
    T *xbuf = nullptr;
    if (incx != 1) {
        xbuf = static_cast<T *>(std::malloc(static_cast<std::size_t>(M) * sizeof(T)));
        if (!xbuf) return false;
        for (int i = 0; i < M; ++i) xbuf[i] = x[static_cast<std::ptrdiff_t>(i) * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        std::ptrdiff_t lo = (static_cast<std::ptrdiff_t>(N) * tid) / nthreads;
        std::ptrdiff_t hi = (static_cast<std::ptrdiff_t>(N) * (tid + 1)) / nthreads;
        for (std::ptrdiff_t j = lo; j < hi; ++j) {
            std::ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;
            std::ptrdiff_t i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            std::ptrdiff_t k = KU - j;
            const T *col = &A_(k + i_lo, j);
            T s = czero;
            if (noconj) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) s = cadd(s, cmul(*col++, xptr[i]));
            else        for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) s = cadd(s, cmul(cconj(*col++), xptr[i]));
            y[j * incy] = cadd(y[j * incy], cmul(alpha, s));
        }
    }
    std::free(xbuf);
    return true;
}
#endif /* _OPENMP */

extern "C" void wgbmv_(
    const char *trans,
    const int *m_, const int *n_,
    const int *kl_, const int *ku_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int KL = *kl_, KU = *ku_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char TR = up(trans);
    const int noconj = (TR == 'T');

    if (M == 0 || N == 0 || (cdd_iszero(alpha) && cdd_isone(beta))) return;

    const int leny = (TR == 'N') ? M : N;
    const int lenx = (TR == 'N') ? N : M;

    if (!cdd_isone(beta)) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (cdd_iszero(beta)) for (int i = 0; i < leny; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < leny; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
    if (cdd_iszero(alpha)) return;

    if (TR == 'N') {
#ifdef _OPENMP
        /* NoTrans threads for contiguous AND strided x/y (helper gathers strided x,
         * writes strided y); bit-identical to the serial scatter (ascending-j). */
        if (M >= WGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && wgbmv_n_omp(M, N, KL, KU, alpha, a, lda, x, incx, y, incy))
            return;
#endif
        if (incx == 1 && incy == 1) {
            for (int j = 0; j < N; ++j) {
                const T tmp = cmul(alpha, x[j]);
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) y[i] = cadd(y[i], cmul(tmp, A_(k + i, j)));
            }
        } else {
            int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
            int ky = (incy < 0) ? -(leny - 1) * incy : 0;
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                const T tmp = cmul(alpha, x[jx]);
                int iy = ky;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) {
                    y[iy] = cadd(y[iy], cmul(tmp, A_(k + i, j)));
                    iy += incy;
                }
                jx += incx;
                if (j >= KU) ky += incy;
            }
        }
    } else if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= WGBMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T s = czero;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            if (noconj) for (int i = i_lo; i < i_hi; ++i) s = cadd(s, cmul(A_(k + i, j), x[i]));
            else        for (int i = i_lo; i < i_hi; ++i) s = cadd(s, cmul(cconj(A_(k + i, j)), x[i]));
            y[j] = cadd(y[j], cmul(alpha, s));
        }
    } else {
        /* Strided Trans/ConjTrans gather. */
#ifdef _OPENMP
        if (N >= WGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && wgbmv_t_omp(M, N, KL, KU, alpha, a, lda, x, incx, y, incy, noconj))
            return;
#endif
        int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        int ky = (incy < 0) ? -(leny - 1) * incy : 0;
        int jy = ky;
        for (int j = 0; j < N; ++j) {
            T s = czero;
            int ix = kx;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            if (noconj) {
                for (int i = i_lo; i < i_hi; ++i) {
                    s = cadd(s, cmul(A_(k + i, j), x[ix]));
                    ix += incx;
                }
            } else {
                for (int i = i_lo; i < i_hi; ++i) {
                    s = cadd(s, cmul(cconj(A_(k + i, j)), x[ix]));
                    ix += incx;
                }
            }
            y[jy] = cadd(y[jy], cmul(alpha, s));
            jy += incy;
            if (j >= KU) kx += incx;
        }
    }
}

#undef A_
