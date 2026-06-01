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
        /* 8-way unroll. Per-element x87 op counts are identical at any unroll
         * factor (2 fldt, 1 fmul, 1 faddp, 1 fstpt — no SIMD for fp80), so the
         * only lever is loop-overhead amortization: unrolling by 8 (the
         * epblas-openblas daxpy head) halves the per-element increment/compare/
         * branch cost vs a 4-way body and recovers ~3% over the L1-resident
         * range. Scalar tail handles n % 8. */
        const int m = n % 8;
        for (int i = 0; i < m; ++i) y[i] += alpha * x[i];
        for (int i = m; i < n; i += 8) {
            y[i    ] += alpha * x[i    ];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
            y[i + 4] += alpha * x[i + 4];
            y[i + 5] += alpha * x[i + 5];
            y[i + 6] += alpha * x[i + 6];
            y[i + 7] += alpha * x[i + 7];
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}
