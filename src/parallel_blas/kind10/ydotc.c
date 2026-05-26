/* ydotc — kind10 complex: returns Σ conj(X)·Y.
 *
 * Single accumulator: 2-acc unroll was tried (commit add00f58) to expose
 * ILP and mask x87 fmul latency, but each `_Complex long double` multiply
 * needs ~6 fp80 slots — 2 accs + temp slots overflow the 8-deep x87
 * stack and force fxch/spill (Addendum 1 §kind10 complex). With single
 * acc and pointer-walk, gcc's scheduler produces an inner loop close to
 * gfortran's reference codegen for ZDOTC.
 */
typedef _Complex long double T;

T ydotc_(const int *n_, const T *x, const int *incx_,
         const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return (T)0.0L;
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        const T *xe = x + n;
        for (const T *xp = x, *yp = y; xp < xe; ++xp, ++yp) s += ~*xp * *yp;
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s += ~x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}
