/* iwamax — multifloats complex DD: 1-based argmax(|re|+|im|). */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline bool dd_gt(R a, R b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
inline R cabs1(T const &z) { return fabsdd(z.re) + fabsdd(z.im); }
}

extern "C" int iwamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
    int best = 1;
    R bestv = cabs1(x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        R av = cabs1(x[ix]);
        if (dd_gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}
