/* edot — kind10 real: returns Σ X·Y. */
typedef long double T;

T edot_(const int *n_, const T *x, const int *incx_,
        const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return 0.0L;
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        /* 5-accumulator unroll matches NETLIB DDOT, masks the ~3-cycle
         * x87 fadd latency. */
        T s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L, s4 = 0.0L;
        int i = 0;
        for (; i + 4 < n; i += 5) {
            s0 += x[i    ] * y[i    ];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            s4 += x[i + 4] * y[i + 4];
        }
        s = s0 + s1 + s2 + s3 + s4;
        for (; i < n; ++i) s += x[i] * y[i];
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}
