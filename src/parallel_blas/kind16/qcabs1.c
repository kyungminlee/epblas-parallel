/* qcabs1 — kind16: |re(z)| + |im(z)|.
 *
 * Use __builtin_fabsf128 instead of libquadmath's fabsq — the builtin
 * compiles to a single `pand` masking the IEEE binary128 sign bit;
 * fabsq is a function call (~50 cycles).
 */
#include <quadmath.h>
typedef __complex128 T;
typedef __float128 R;
R qcabs1_(const T *z) {
    return __builtin_fabsf128(__real__ *z) + __builtin_fabsf128(__imag__ *z);
}
