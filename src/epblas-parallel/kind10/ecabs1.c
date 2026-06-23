/* ecabs1 — kind10: |re(z)| + |im(z)| for one complex long double. */
#include <math.h>
typedef _Complex long double TC;
typedef long double R;
/* No integer arguments -> the LP64 and ILP64 ABIs are identical; the _64_ twin
 * forwards (tail-call) so the archive still exports both symbols uniformly. */
R ecabs1_(const TC *z) { return fabsl(__real__ *z) + fabsl(__imag__ *z); }
R ecabs1_64_(const TC *z) { return ecabs1_(z); }
