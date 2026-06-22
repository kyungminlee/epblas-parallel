#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* eswap — kind10 real: swap X ↔ Y. */
typedef long double T;

/* eswap has two unit-stride kernels because the serial and threaded regimes
 * have OPPOSITE optima (each measured to within ~1% and cross-checked: using
 * either kernel in the other's regime reproduces that regime's gap).
 *
 * eswap_unit (serial path): gfortran netlib dswap shape — 3-way unrolled,
 * INTERLEAVED per element (t=x; x=y; y=t). Keeps a tight load->store chain at
 * x87 stack-depth 2. This is the latency-bound regime; the phase-separated
 * 4-way form below loses a stable ~3% here (par/mig 1.03 at every size). With
 * the interleaved form par/mig is 1.01@1024, 0.98@65536, 0.99@1M (par now
 * beats mig — it was NOT a gfortran-codegen floor, just the wrong shape). */
static void eswap_unit(ptrdiff_t n, T *x, T *y)
{
    ptrdiff_t i, m = n % 3;
    for (i = 0; i < m; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
    for (; i < n; i += 3) {
        T t0 = x[i+0]; x[i+0] = y[i+0]; y[i+0] = t0;
        T t1 = x[i+1]; x[i+1] = y[i+1]; y[i+1] = t1;
        T t2 = x[i+2]; x[i+2] = y[i+2]; y[i+2] = t2;
    }
}

#ifdef _OPENMP
/* eswap_slice (per-thread OMP slices): OpenBLAS dswap shape — 4-way unrolled
 * and PHASE-SEPARATED (load all four x's, overwrite x from y, then drain the
 * temps into y). Deepening the x87 stack to 4 lets several loads/stores stay in
 * flight, which wins in the bandwidth-bound threaded regime: it closes a stable
 * ~9% par/ob gap at N=65536 OMP4. Using the interleaved serial kernel here
 * instead reopens that gap (par/ob 1.09 @65536 OMP4). */
static void eswap_slice(ptrdiff_t n, T *x, T *y)
{
    ptrdiff_t i, n1 = n & -4;
    for (i = 0; i < n1; i += 4) {
        T t0 = x[i+0], t1 = x[i+1], t2 = x[i+2], t3 = x[i+3];
        x[i+0] = y[i+0]; x[i+1] = y[i+1]; x[i+2] = y[i+2]; x[i+3] = y[i+3];
        y[i+0] = t0; y[i+1] = t1; y[i+2] = t2; y[i+3] = t3;
    }
    for (; i < n; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
}

/* Threaded unit-stride SWAP — same cache-bandwidth rationale as eaxpy_omp
 * (see eaxpy.c). swap is the heaviest RMW (2 reads + 2 writes/elem). Threshold
 * is set by par4<=ob4 (ob keeps swap serial at small N, so par's 4-thread time
 * must beat ob's *serial* time). Measured under iomp5: par4/ob4 1.34@1024,
 * 1.12@2048, 1.03@3072, then 0.96@4096 and 0.77@6144 — break-even ~4096, stay
 * serial through 3072. */
#define ESWAP_OMP_MIN 3072
static bool eswap_omp(ptrdiff_t n, T *x, T *y)
{
    if (n <= ESWAP_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) eswap_slice(hi - lo, x + lo, y + lo);
    }
    return 1;
}
#endif

static void eswap_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (eswap_omp(n, x, y)) return;
#endif
        eswap_unit(n, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_SWAP(eswap, T)
