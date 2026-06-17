/* msyr — multifloats real DD symmetric rank-1 update. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
#define MSYR_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void msyr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    T *a, const int *lda_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, lda = *lda_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || dd_iszero(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= MSYR_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const int use_omp = 0;
#endif
        const std::ptrdiff_t n = N;
        /* The serial and threaded paths want different shapes, so split them
         * at C++ source level instead of via `#pragma omp parallel for
         * if(use_omp)` — the `if()` clause outlines the loop body
         * unconditionally into a GOMP closure, forcing even the OMP=1 path
         * through the outlined function, whose codegen lost ~3-5% on the
         * serial UPPER triangle (par/ob ~1.03-1.06).
         *
         * Serial path: UPLO hoisted OUT of the column loop into two
         * specialized loops, ptrdiff_t indices, `+=` — byte-for-byte the ob
         * clone, so the inner address arithmetic strength-reduces to a pure
         * pointer walk with no per-element sign extension.
         *
         * Threaded path: a single column loop with the branch inside (the
         * `#pragma omp parallel for` must bind directly to a `for`); the
         * per-column UPLO test is negligible against the threading win, and
         * the outlined-closure cost is irrelevant once threads are live.
         * static,1 cyclic interleave balances the triangular column skew
         * (column j writes j+1 / N-j elems); full storage → columns lda
         * apart, no false sharing. */
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static, 1)
#endif
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T t = alpha * x[j];
                if (dd_iszero(t)) continue;
                T *aj = &A_(0, j);
                if (UPLO == 'L') for (std::ptrdiff_t i = j; i < n; ++i) aj[i] += t * x[i];
                else             for (std::ptrdiff_t i = 0; i <= j; ++i) aj[i] += t * x[i];
            }
        } else if (UPLO == 'L') {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T t = alpha * x[j];
                if (dd_iszero(t)) continue;
                T *aj = &A_(0, j);
                for (std::ptrdiff_t i = j; i < n; ++i) aj[i] += t * x[i];
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T t = alpha * x[j];
                if (dd_iszero(t)) continue;
                T *aj = &A_(0, j);
                for (std::ptrdiff_t i = 0; i <= j; ++i) aj[i] += t * x[i];
            }
        }
    } else {
        /* General-stride fallback. Hoist the column pointer and walk x with a
         * running index (ix += incx) instead of recomputing A_(i,j) and
         * x[kx+i*incx] each element — mirrors the contiguous path / ob clone.
         * kx keeps the reference negative-incx start (first logical element at
         * the high-memory end). Rare path, kept serial. */
        const std::ptrdiff_t n = N;
        const std::ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const T t = alpha * x[kx + j * incx];
            if (dd_iszero(t)) continue;
            T *aj = &A_(0, j);
            if (UPLO == 'L') {
                std::ptrdiff_t ix = kx + j * incx;
                for (std::ptrdiff_t i = j; i < n; ++i) { aj[i] += t * x[ix]; ix += incx; }
            } else {
                std::ptrdiff_t ix = kx;
                for (std::ptrdiff_t i = 0; i <= j; ++i) { aj[i] += t * x[ix]; ix += incx; }
            }
        }
    }
}

#undef A_
