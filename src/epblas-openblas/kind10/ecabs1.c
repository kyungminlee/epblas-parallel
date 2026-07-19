/*
 * ecabs1 — kind10 port of OpenBLAS dcabs1.  |Re(z)| + |Im(z)|.
 */
typedef _Complex long double C;
typedef long double T;

static inline T ldabs(T x) { return __builtin_fabsl(x); }  /* branchless x87 fabs */

T ecabs1_(const C *Z)
{
    const T *p = (const T *)Z;
    return ldabs(p[0]) + ldabs(p[1]);
}
