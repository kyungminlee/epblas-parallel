/* eaxpy — kind10 real: Y := α·X + Y. */
typedef long double T;

void eaxpy_(const int *n_, const T *alpha_,
            const T *x, const int *incx_,
            T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 0.0L) return;
    if (incx == 1 && incy == 1) {
        /* 4-way unroll matches NETLIB DAXPY. */
        const int m = n % 4;
        for (int i = 0; i < m; ++i) y[i] += alpha * x[i];
        for (int i = m; i < n; i += 4) {
            y[i    ] += alpha * x[i    ];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}
