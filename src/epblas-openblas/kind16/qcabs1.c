/*
 * qcabs1 — kind16 port of OpenBLAS dcabs1.  |Re(z)| + |Im(z)|.
 */
#include <quadmath.h>
typedef __complex128 C;
typedef __float128 T;

static inline T q_abs(T x) { return __builtin_fabsf128(x); }

T qcabs1_(const C *Z)
{
    const T *p = (const T *)Z;
    return q_abs(p[0]) + q_abs(p[1]);
}
