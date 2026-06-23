/* erotg — kind10 real Givens generator. */
#include <math.h>
typedef long double TR;

void erotg_(TR *a, TR *b, TR *c, TR *s)
{
    TR anorm = fabsl(*a), bnorm = fabsl(*b);
    if (bnorm == 0.0L) { *c = 1.0L; *s = 0.0L; *b = 0.0L; return; }
    if (anorm == 0.0L) { *c = 0.0L; *s = 1.0L; *a = *b; *b = 1.0L; return; }
    TR scale = anorm > bnorm ? anorm : bnorm;
    TR ax = *a / scale, bx = *b / scale;
    TR r = scale * sqrtl(ax * ax + bx * bx);
    if (*a < 0.0L) r = -r;
    *c = *a / r;
    *s = *b / r;
    TR z = (anorm > bnorm) ? *s : 1.0L / *c;
    *a = r;
    *b = z;
}

/* No integer arguments -> LP64/ILP64 ABIs identical; _64_ twin tail-calls. */
void erotg_64_(TR *a, TR *b, TR *c, TR *s) { erotg_(a, b, c, s); }
