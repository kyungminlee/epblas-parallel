/* qrotg — kind16 real Givens generator. */
#include <quadmath.h>
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;

void qrotg_(T *a, T *b, T *c, T *s)
{
    T anorm = fabsq(*a), bnorm = fabsq(*b);
    if (bnorm == 0.0Q) { *c = 1.0Q; *s = 0.0Q; *b = 0.0Q; return; }
    if (anorm == 0.0Q) { *c = 0.0Q; *s = 1.0Q; *a = *b; *b = 1.0Q; return; }
    T scale = anorm > bnorm ? anorm : bnorm;
    T ax = *a / scale, bx = *b / scale;
    T r = scale * sqrtq(ax * ax + bx * bx);
    if (*a < 0.0Q) r = -r;
    *c = *a / r;
    *s = *b / r;
    T z = (anorm > bnorm) ? *s : 1.0Q / *c;
    *a = r;
    *b = z;
}
