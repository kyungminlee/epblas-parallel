/* mrotm — multifloats real DD: apply modified Givens rotation.
 * H · (X, Y) determined by flag in dparam[0] ∈ {-2, -1, 0, +1}.
 */
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
inline T dd_neg(T x) { return T{-x.limbs[0], -x.limbs[1]}; }
}

extern "C" void mrotm_(const int *n_,
                       T *x, const int *incx_,
                       T *y, const int *incy_,
                       const T *dparam)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T flag = dparam[0];
    /* flag == -2: identity, do nothing */
    T two{2.0, 0.0};
    if (n <= 0 || dd_eq0(flag + two)) return;

    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];

    auto step = [&](T &xi, T &yi) {
        T w = xi, z = yi;
        if (dd_lt0(flag)) {            /* flag = -1: full H matrix */
            xi = w * h11 + z * h12;
            yi = w * h21 + z * h22;
        } else if (dd_eq0(flag)) {     /* flag = 0: H = [1 h12; h21 1] */
            xi = w + z * h12;
            yi = w * h21 + z;
        } else {                       /* flag = +1: H = [h11 1; -1 h22] */
            xi = w * h11 + z;
            yi = dd_neg(w) + h22 * z;
        }
    };

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        /* Disjoint (x[i], y[i]) updates → thread the loop directly. Scalar
         * (no SIMD path); DD math keeps it compute-bound. */
        if (n > MROTM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            int nthreads = blas_omp_max_threads();
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (int i = 0; i < n; ++i) step(x[i], y[i]);
            return;
        }
#endif
        for (int i = 0; i < n; ++i) step(x[i], y[i]);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { step(x[ix], y[iy]); ix += incx; iy += incy; }
    }
}
