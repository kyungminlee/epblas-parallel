/* yaxpy — kind10 complex: Y := α·X + Y. */
typedef _Complex long double T;

void yaxpy_(const int *n_, const T *alpha_,
            const T *x, const int *incx_,
            T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    if (n <= 0) return;
    if (alpha == (T)0.0L) return;
    if (incx == 1 && incy == 1) {
        const T *xe = x + n;
        T *yp = y;
        for (const T *xp = x; xp < xe; ++xp, ++yp) *yp += alpha * (*xp);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}
