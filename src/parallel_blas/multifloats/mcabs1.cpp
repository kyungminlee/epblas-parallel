/* mcabs1 — multifloats: |re(z)| + |im(z)| for a single complex DD. */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

extern "C" R mcabs1_(const T *z_)
{
    const T z = *z_;
    return fabsdd(z.re) + fabsdd(z.im);
}
