/* yerot — kind10: complex Givens with real c, s. */
typedef _Complex long double T;
typedef long double R;

void yerot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
            const R *c_, const R *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const R c = *c_, s = *s_;
    if (n <= 0) return;
    /* Real coefficients on complex data: treat each complex element as two
     * independent reals and run the plain real rotation over them.  The
     * contiguous case becomes one flat 2n real loop (the epblas-openblas
     * shape) — load-first, no _Complex multiply, accumulators stay on the
     * x87 register stack.  Same ops in the same order — bit-identical. */
    if (incx == 1 && incy == 1) {
        R *px = (R *)x, *py = (R *)y;
        const long two_n = 2L * n;
        for (long i = 0; i < two_n; ++i) {
            R xi = px[i], yi = py[i];
            px[i] = c * xi + s * yi;
            py[i] = c * yi - s * xi;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) {
            R *px = (R *)&x[ix], *py = (R *)&y[iy];
            R xr = px[0], xi = px[1], yr = py[0], yi = py[1];
            px[0] = c * xr + s * yr; px[1] = c * xi + s * yi;
            py[0] = c * yr - s * xr; py[1] = c * yi - s * xi;
            ix += incx; iy += incy;
        }
    }
}
