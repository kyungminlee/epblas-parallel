/*
 * mcabs1 — multifloats DD port of OpenBLAS dcabs1.  |Re(z)| + |Im(z)|.
 *
 * Faithful retype of kind10/ecabs1.c: the complex element is read through a
 * float64x2* alias (complex64x2 = two float64x2 limbs).
 */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

static inline T ldabs(T x) { return x < 0.0 ? -x : x; }

extern "C" T mcabs1_(const mf::complex64x2 *Z)
{
    const T *p = reinterpret_cast<const T *>(Z);
    return ldabs(p[0]) + ldabs(p[1]);
}
