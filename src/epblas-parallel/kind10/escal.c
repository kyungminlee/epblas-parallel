#include <stddef.h>
/* escal — kind10 real: X := α · X. x87 fp80, no SIMD path.
 *
 * Inner loop is 5-way unrolled to match the NETLIB DSCAL reference.
 * Each x87 `fmulp` has ~3-cycle latency; unrolling lets the OOO core
 * dispatch independent fldt/fmul/fstpt sequences in parallel since all
 * five mults share alpha but touch distinct x[i]. Plain `for(i) x[i] *=
 * alpha` lost ~25% vs migrated at every size above N=128 without it.
 */
typedef long double T;

void escal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 1.0L) return;
    if (incx == 1) {
        const ptrdiff_t m = n % 5;
        for (ptrdiff_t i = 0; i < m; ++i) x[i] *= alpha;
        for (ptrdiff_t i = m; i < n; i += 5) {
            x[i    ] *= alpha;
            x[i + 1] *= alpha;
            x[i + 2] *= alpha;
            x[i + 3] *= alpha;
            x[i + 4] *= alpha;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}
