/* whpr2 — multifloats complex DD Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 *
 * Columns independent → OMP over j. The packed triangular output makes a
 * contiguous static block hand one thread the heavy triangle end (par caps
 * at ~2.3x on 4 cores); cyclic schedule(static,1) interleaves short and long
 * columns across the team, balancing the skew symmetrically for both UPLO
 * (mirrors the proven kind10 yhpr2). The per-column body is carved into a
 * noinline helper so the inner loop keeps clean register allocation and the
 * serial + threaded paths share one tight loop (inlining into the omp region
 * spills the kept-resident column scalars and loses the UPPER triangle).
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
#define WHPR2_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

/* Per-column rank-2 update, carved out noinline so the inner loop compiles
 * with clean register allocation (see file header). The Hermitian diagonal
 * is forced real: the off-diagonal run plus the single real diagonal write. */
__attribute__((noinline))
void whpr2_col_upper(int j, T t1, T t2, const T *x, const T *y, T *ap) {
    T *c = ap + static_cast<std::size_t>(j) * (j + 1) / 2;
    for (int i = 0; i < j; ++i)
        c[i] = cadd(c[i], cadd(cmul(x[i], t1), cmul(y[i], t2)));
    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    c[j] = T{ c[j].re + prod.re, rzero };
}

__attribute__((noinline))
void whpr2_col_lower(int j, int N, T t1, T t2, const T *x, const T *y, T *ap) {
    /* Pre-advance the off-diagonal bases so the loop runs 0-based over one
     * induction variable indexing three pointers — the tight form gcc picks
     * for the upper helper. Diagonal last, on a clean stack. */
    const int mo = N - j - 1;
    T *c0 = ap + (static_cast<std::size_t>(j) * N - static_cast<std::size_t>(j) * (j - 1) / 2);
    T *c = c0 + 1;
    const T *xc = x + j + 1, *yc = y + j + 1;
    for (int i = 0; i < mo; ++i)
        c[i] = cadd(c[i], cadd(cmul(xc[i], t1), cmul(yc[i], t2)));
    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    c0[0] = T{ c0[0].re + prod.re, rzero };
}
}

extern "C" void whpr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || cdd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= WHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[j]) || !cdd_iszero(y[j]))
                    whpr2_col_upper(j, cmul(alpha, cconj(y[j])),
                                    cconj(cmul(alpha, x[j])), x, y, ap);
                else {
                    const std::size_t kk = static_cast<std::size_t>(j) * (j + 1) / 2;
                    ap[kk + j] = T{ ap[kk + j].re, rzero };
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= WHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[j]) || !cdd_iszero(y[j]))
                    whpr2_col_lower(j, N, cmul(alpha, cconj(y[j])),
                                    cconj(cmul(alpha, x[j])), x, y, ap);
                else {
                    const std::size_t kk = static_cast<std::size_t>(j) * N
                                         - static_cast<std::size_t>(j) * (j - 1) / 2;
                    ap[kk] = T{ ap[kk].re, rzero };
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        int kk = 0;
        int jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[jx]) || !cdd_iszero(y[jy])) {
                    const T t1 = cmul(alpha, cconj(y[jy]));
                    const T t2 = cconj(cmul(alpha, x[jx]));
                    int ix = kx, iy = ky;
                    for (int k = kk; k < kk + j; ++k) {
                        ap[k] = cadd(ap[k], cadd(cmul(x[ix], t1), cmul(y[iy], t2)));
                        ix += incx; iy += incy;
                    }
                    const T prod = cadd(cmul(x[jx], t1), cmul(y[jy], t2));
                    ap[kk + j] = T{ ap[kk + j].re + prod.re, rzero };
                } else {
                    ap[kk + j] = T{ ap[kk + j].re, rzero };
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[jx]) || !cdd_iszero(y[jy])) {
                    const T t1 = cmul(alpha, cconj(y[jy]));
                    const T t2 = cconj(cmul(alpha, x[jx]));
                    const T prod = cadd(cmul(x[jx], t1), cmul(y[jy], t2));
                    ap[kk] = T{ ap[kk].re + prod.re, rzero };
                    int ix = jx, iy = jy;
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx; iy += incy;
                        ap[k] = cadd(ap[k], cadd(cmul(x[ix], t1), cmul(y[iy], t2)));
                    }
                } else {
                    ap[kk] = T{ ap[kk].re, rzero };
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
