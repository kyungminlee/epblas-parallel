/*
 * qcabs1 — kind16 port of OpenBLAS dcabs1.  |Re(z)| + |Im(z)|.
 */
#include <quadmath.h>
typedef __complex128 C;
typedef __float128 T;

static inline T ldabs(T x) { return x < 0 ? -x : x; }

T qcabs1_(const C *Z)
{
    const T *p = (const T *)Z;
    return ldabs(p[0]) + ldabs(p[1]);
}
