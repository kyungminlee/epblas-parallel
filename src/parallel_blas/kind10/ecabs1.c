/* ecabs1 — kind10: |re(z)| + |im(z)| for one complex long double. */
#include <math.h>
typedef _Complex long double T;
typedef long double R;
R ecabs1_(const T *z) { return fabsl(__real__ *z) + fabsl(__imag__ *z); }
