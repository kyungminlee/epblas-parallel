/* erot — kind10 real Givens rotation. */
typedef long double T;

void erot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
           const T *c_, const T *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) {
            T tx = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = tx;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) {
            T tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}
