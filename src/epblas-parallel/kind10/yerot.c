/* yerot — kind10: complex Givens with real c, s. */
typedef _Complex long double T;
typedef long double R;

void yerot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
            const R *c_, const R *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const R c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        T *xe = x + n;
        T *xp = x, *yp = y;
        for (; xp < xe; ++xp, ++yp) {
            T tx = c * (*xp) + s * (*yp);
            *yp = c * (*yp) - s * (*xp);
            *xp = tx;
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
