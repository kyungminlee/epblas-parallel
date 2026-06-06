/* wtrmv — multifloats complex DD triangular matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WTRMV_OMP_MIN 256
#define WTRMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* Row-gather: x := A*x / A^T*x / A^H*x, in-place dense triangular. Each output
 * row r is an independent dot once original x is copied off. NoTrans reads matrix
 * row r (A_(r,c), strided by lda, elements direct); Trans/ConjTrans reads column
 * r (A_(c,r), contiguous, conjugated when conj). Diagonal scales x[r] when
 * non-unit (conjugated when conj). xin is the contiguous copy, xout the
 * destination. Mirrors the serial accumulation order → within DD fuzz tol; the
 * serial path stays bit-exact. */
static void wtrmv_rowgather(bool upper, bool trans, bool conj, bool nounit,
                            int n, int lo, int hi,
                            const T *a, std::size_t lda,
                            const T *xin, T *xout, int incx)
{
    for (int r = lo; r < hi; ++r) {
        T diagc = A_(r, r); if (conj) diagc = cconj(diagc);
        T s = nounit ? cmul(diagc, xin[r]) : xin[r];
        if (!trans) {
            if (upper) for (int c = r + 1; c < n; ++c) s = cadd(s, cmul(A_(r, c), xin[c]));
            else       for (int c = 0; c < r; ++c)     s = cadd(s, cmul(A_(r, c), xin[c]));
        } else {
            const T *aj = &A_(0, r);
            if (upper) {
                for (int c = 0; c < r; ++c) {
                    T e = aj[c]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[c]));
                }
            } else {
                for (int c = r + 1; c < n; ++c) {
                    T e = aj[c]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[c]));
                }
            }
        }
        xout[(std::ptrdiff_t)r * incx] = s;
    }
}

/* Threaded in-place complex dense triangular matvec. Copy x to contiguous,
 * partition the output rows, write back. Returns true if handled. */
__attribute__((noinline)) static bool wtrmv_omp(
    bool upper, bool trans, bool conj, bool nounit, int n,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < WTRMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTRMV_MAX_CPUS) nthreads = WTRMV_MAX_CPUS;

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        wtrmv_rowgather(upper, trans, conj, nounit, n, lo, hi, a, lda, xbuf, xbase, incx);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void wtrmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *a, const int *lda_,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR   = up(trans);
    const char DIAG = up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= WTRMV_OMP_MIN && blas_omp_max_threads() > 1
        && wtrmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[j];
                    if (!cdd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = j + 1; i < N; ++i) x[i] = cadd(x[i], cmul(temp, aj[i]));
                    }
                    if (nounit) x[j] = cmul(x[j], A_(j, j));
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[j];
                    if (!cdd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = 0; i < j; ++i) x[i] = cadd(x[i], cmul(temp, aj[i]));
                    }
                    if (nounit) x[j] = cmul(x[j], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = j + 1; i < N; ++i) temp = cadd(temp, cmul(cconj(aj[i]), x[i]));
                    } else {
                        for (int i = j + 1; i < N; ++i) temp = cadd(temp, cmul(aj[i], x[i]));
                    }
                    x[j] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = 0; i < j; ++i) temp = cadd(temp, cmul(cconj(aj[i]), x[i]));
                    } else {
                        for (int i = 0; i < j; ++i) temp = cadd(temp, cmul(aj[i], x[i]));
                    }
                    x[j] = temp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[kx + j * incx];
                    if (!cdd_iszero(temp))
                        for (int i = j + 1; i < N; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (!cdd_iszero(temp))
                        for (int i = 0; i < j; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (int i = j + 1; i < N; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (int i = 0; i < j; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

#undef A_
