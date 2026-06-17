/* mf_dotkernel.h — shared unit-stride complex-DD dot kernels.
 *
 * The AVX2 Bailey-wide dot (4 cells/iter, 3-double accumulators) lives in
 * wdotu.cpp / wdotc.cpp. Exposing the inner unit-stride kernels lets packed and
 * banded triangular matvecs (e.g. wtpmv contiguous Trans) reuse the vectorized
 * column dot instead of a scalar cmul/cadd loop. The kernels reorder summation
 * relative to a sequential reference, so callers that adopt them trade bit-exact
 * accumulation for within-fuzz-tol results (and typically higher accuracy). */
#ifndef EPBLAS_PARALLEL_MF_DOTKERNEL_H
#define EPBLAS_PARALLEL_MF_DOTKERNEL_H

#include <multifloats.h>

namespace mfdot {
/* Sum x[i]*y[i] over a contiguous unit-stride range (unconjugated). */
multifloats::complex64x2 wdotu_unit(int n, const multifloats::complex64x2 *x,
                                    const multifloats::complex64x2 *y);
/* Sum conj(x[i])*y[i] over a contiguous unit-stride range. */
multifloats::complex64x2 wdotc_unit(int n, const multifloats::complex64x2 *x,
                                    const multifloats::complex64x2 *y);
}

#endif /* EPBLAS_PARALLEL_MF_DOTKERNEL_H */
