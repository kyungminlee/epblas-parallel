/*
 * mcabs1 — multifloats DD port of OpenBLAS dcabs1.  |Re(z)| + |Im(z)|.
 *
 * Faithful retype of kind10/ecabs1.c: the complex element is read through a
 * float64x2* alias (complex64x2 = two float64x2 limbs).
 */
#include <type_traits>
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

static inline T dd_abs(T x) { return mf::fabs(x); }

/* The limb-alias read below (and the C-heap complex buffers elsewhere in
 * this leg) rely on complex64x2 being two trivially copyable float64x2
 * limbs; pin that layout/lifetime assumption once, here: */
static_assert(std::is_trivially_copyable<mf::complex64x2>::value &&
              sizeof(mf::complex64x2) == 2 * sizeof(T),
              "complex64x2 must stay two trivially-copyable float64x2 limbs");

extern "C" T mcabs1_(const mf::complex64x2 *Z)
{
    const T *p = reinterpret_cast<const T *>(Z);
    return dd_abs(p[0]) + dd_abs(p[1]);
}
