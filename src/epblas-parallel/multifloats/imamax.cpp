/* imamax — multifloats real DD: 1-based argmax(|X|). */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_gt(T a, T b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
}

extern "C" int imamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
    int best = 1;
    T bestv = fabsdd(x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        T av = fabsdd(x[ix]);
        if (dd_gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}
