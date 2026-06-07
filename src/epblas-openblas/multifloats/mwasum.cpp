/*
 * mwasum — multifloats DD port of OpenBLAS dzasum.
 *
 * sum( |Re(z[i])| + |Im(z[i])| ).  Real return, complex input.
 * Faithful retype of kind10/eyasum.c: the complex stream is read through a
 * float64x2* alias (complex64x2 = two float64x2 limbs). Bails on n<=0 or
 * incx<=0.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

#define MULTI_THREAD_MINIMAL 10000

static inline T ldabs(T x) { return x < 0.0 ? -x : x; }

static T asum_kernel(std::ptrdiff_t n, const mf::complex64x2 *x, std::ptrdiff_t incx)
{
    T s = 0.0;
    const T *p;
    if (incx == 1) {
        p = reinterpret_cast<const T *>(x);
        for (std::ptrdiff_t i = 0; i < 2*n; i += 2) {
            s += ldabs(p[i]) + ldabs(p[i+1]);
        }
        return s;
    }
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        p = reinterpret_cast<const T *>(x + i*incx);
        s += ldabs(p[0]) + ldabs(p[1]);
    }
    return s;
}

extern "C" T mwasum_(const int *N, const mf::complex64x2 *x, const int *INCX)
{
    std::ptrdiff_t n    = (std::ptrdiff_t)(*N);
    std::ptrdiff_t incx = (std::ptrdiff_t)(*INCX);
    if (n <= 0 || incx <= 0) return T(0.0);

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            if (nthreads > 64) nthreads = 64;
            T partial[64] = {};
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                std::ptrdiff_t chunk = (n + nth - 1) / nth;
                std::ptrdiff_t start = (std::ptrdiff_t)tid * chunk;
                std::ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end)
                    partial[tid] = asum_kernel(end - start,
                                               x + start * incx, incx);
            }
            T s = 0.0;
            for (int i = 0; i < nthreads; ++i) s += partial[i];
            return s;
        }
    }
#endif
    return asum_kernel(n, x, incx);
}
