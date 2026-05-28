/* ydotu — kind10 complex: returns Σ X·Y (no conjugate).
 *
 * Single accumulator: see ydotc for rationale (x87 stack pressure
 * from 2-acc unroll, Addendum 1 §kind10 complex). */
typedef _Complex long double T;

T ydotu_(const int *n_, const T *x, const int *incx_,
         const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return (T)0.0L;
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        const T *xe = x + n;
        for (const T *xp = x, *yp = y; xp < xe; ++xp, ++yp) s += *xp * *yp;
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}
