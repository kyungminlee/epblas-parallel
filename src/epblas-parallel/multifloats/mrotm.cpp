/* mrotm — multifloats real DD: apply modified Givens rotation.
 * H · (X, Y) determined by flag in dparam[0] ∈ {-2, -1, 0, +1}.
 *
 * flag is loop-invariant, so it is unswitched OUT of the element loop into three
 * flag-specific kernels (source-level unswitching). Findings rule 7 — do NOT
 * fold back to one inner loop with a per-element branch; gcc then loses the
 * unswitch and emits ~3x the stores (the prior lambda-per-element form cost
 * ~7-10% serially vs ob).
 */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MROTM_OMP_MIN 1024
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_lt0(T x) { return x.limbs[0] < 0.0 || (x.limbs[0] == 0.0 && x.limbs[1] < 0.0); }
inline bool dd_eq0(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }

void rotm_neg(std::ptrdiff_t lo, std::ptrdiff_t hi, T *x, T *y,
              T h11, T h12, T h21, T h22) {
    for (std::ptrdiff_t i = lo; i < hi; ++i) {
        T w = x[i], z = y[i];
        x[i] = w * h11 + z * h12;
        y[i] = w * h21 + z * h22;
    }
}
void rotm_zero(std::ptrdiff_t lo, std::ptrdiff_t hi, T *x, T *y, T h12, T h21) {
    for (std::ptrdiff_t i = lo; i < hi; ++i) {
        T w = x[i], z = y[i];
        x[i] = w + z * h12;
        y[i] = w * h21 + z;
    }
}
void rotm_pos(std::ptrdiff_t lo, std::ptrdiff_t hi, T *x, T *y, T h11, T h22) {
    for (std::ptrdiff_t i = lo; i < hi; ++i) {
        T w = x[i], z = y[i];
        x[i] = w * h11 + z;
        y[i] = T{-w.limbs[0], -w.limbs[1]} + h22 * z;
    }
}
}

extern "C" void mrotm_(const int *n_,
                       T *x, const int *incx_,
                       T *y, const int *incy_,
                       const T *dparam)
{
    const std::ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    const T flag = dparam[0];
    /* flag == -2: identity, do nothing */
    if (n <= 0 || dd_eq0(flag + T{2.0, 0.0})) return;

    const bool neg = dd_lt0(flag), zero = dd_eq0(flag);
    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (n > MROTM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            int nthreads = blas_omp_max_threads();
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                std::ptrdiff_t chunk = (n + nth - 1) / nth;
                std::ptrdiff_t s = (std::ptrdiff_t)tid * chunk;
                std::ptrdiff_t e = s + chunk;
                if (e > n) e = n;
                if (s < e) {
                    if (neg)       rotm_neg(s, e, x, y, h11, h12, h21, h22);
                    else if (zero) rotm_zero(s, e, x, y, h12, h21);
                    else           rotm_pos(s, e, x, y, h11, h22);
                }
            }
            return;
        }
#endif
        if (neg)       rotm_neg(0, n, x, y, h11, h12, h21, h22);
        else if (zero) rotm_zero(0, n, x, y, h12, h21);
        else           rotm_pos(0, n, x, y, h11, h22);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        if (neg) {
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                T w = x[ix], z = y[iy];
                x[ix] = w * h11 + z * h12;
                y[iy] = w * h21 + z * h22;
                ix += incx; iy += incy;
            }
        } else if (zero) {
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                T w = x[ix], z = y[iy];
                x[ix] = w + z * h12;
                y[iy] = w * h21 + z;
                ix += incx; iy += incy;
            }
        } else {
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                T w = x[ix], z = y[iy];
                x[ix] = w * h11 + z;
                y[iy] = T{-w.limbs[0], -w.limbs[1]} + h22 * z;
                ix += incx; iy += incy;
            }
        }
    }
}
