/* erotm — kind10 real: apply modified Givens.
 *
 * Imag-part products written with the just-loaded operand (z) first so
 * gcc's x87 backend emits the multiply against top-of-stack and saves
 * a fxch — same pattern as yscal (Addendum 17). The flag-unswitched
 * paths each shrink from 14 insns + 2 fxch to 12 insns + 1 fxch in
 * the flag<0 branch. */
typedef long double T;

static inline void step(const T flag, const T h11, const T h12, const T h21, const T h22,
                        T *xi, T *yi)
{
    T w = *xi, z = *yi;
    if (flag < 0.0L)        { *xi = w * h11 + z * h12; *yi = z * h22 + w * h21; }
    else if (flag == 0.0L)  { *xi = w + z * h12;       *yi = z + w * h21; }
    else                    { *xi = w * h11 + z;       *yi = z * h22 - w; }
}

void erotm_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
            const T *dparam)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T flag = dparam[0];
    if (n <= 0 || flag == -2.0L) return;
    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) step(flag, h11, h12, h21, h22, &x[i], &y[i]);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { step(flag, h11, h12, h21, h22, &x[ix], &y[iy]);
                                       ix += incx; iy += incy; }
    }
}
