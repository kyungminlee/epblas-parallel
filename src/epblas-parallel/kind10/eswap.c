/* eswap — kind10 real: swap X ↔ Y. */
typedef long double T;

void eswap_(const int *n_, T *x, const int *incx_, T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        const int m = n % 3;
        for (int i = 0; i < m; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
        for (int i = m; i < n; i += 3) {
            T t0 = x[i    ]; x[i    ] = y[i    ]; y[i    ] = t0;
            T t1 = x[i + 1]; x[i + 1] = y[i + 1]; y[i + 1] = t1;
            T t2 = x[i + 2]; x[i + 2] = y[i + 2]; y[i + 2] = t2;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}
