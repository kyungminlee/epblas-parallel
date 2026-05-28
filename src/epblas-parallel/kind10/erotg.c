/* erotg — kind10 real Givens generator. */
#include <math.h>
typedef long double T;

void erotg_(T *a, T *b, T *c, T *s)
{
    T anorm = fabsl(*a), bnorm = fabsl(*b);
    if (bnorm == 0.0L) { *c = 1.0L; *s = 0.0L; *b = 0.0L; return; }
    if (anorm == 0.0L) { *c = 0.0L; *s = 1.0L; *a = *b; *b = 1.0L; return; }
    T scale = anorm > bnorm ? anorm : bnorm;
    T ax = *a / scale, bx = *b / scale;
    T r = scale * sqrtl(ax * ax + bx * bx);
    if (*a < 0.0L) r = -r;
    *c = *a / r;
    *s = *b / r;
    T z = (anorm > bnorm) ? *s : 1.0L / *c;
    *a = r;
    *b = z;
}
