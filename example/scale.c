/* Minimal downstream consumer of the kind10 (long double) overlay.
 *
 * Calls the Fortran-ABI entry point escal_ (X := alpha * X) exposed by the
 * eyblas composite, then prints the linked library version. Build it with the
 * CMakeLists.txt beside this file against an installed epblas-parallel; see
 * doc/user/usage.md. */
#include <stdio.h>
#include <epblas-parallel/version.h>

/* Fortran-ABI BLAS scal: every argument is passed by reference, integers are
 * default (32-bit) INTEGER. The overlay supplies escal_ for REAL(KIND=10). */
extern void escal_(const int *n, const long double *alpha,
                   long double *x, const int *incx);

int main(void)
{
    enum { N = 5 };
    long double x[N] = {1.0L, 2.0L, 3.0L, 4.0L, 5.0L};
    long double alpha = 2.0L;
    int n = N, incx = 1;

    escal_(&n, &alpha, x, &incx);

    printf("epblas-parallel %s — escal(alpha=2) result:\n",
           EPBLAS_PARALLEL_VERSION_STRING);
    for (int i = 0; i < N; ++i)
        printf("  x[%d] = %.1Lf\n", i, x[i]);
    return 0;
}
