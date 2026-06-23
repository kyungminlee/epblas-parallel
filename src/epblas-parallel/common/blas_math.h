/*
 * blas_math.h — generic integer micro-helpers shared across every kind tree
 * (kind10, kind16, multifloats). These are pure, type-uniform arithmetic on
 * `ptrdiff_t` sizes/strides — the same handful of one-liners that the L3
 * blockers and packers had each been copying file-locally (`round_up`, `imin`).
 *
 * Homed here so there is a single definition, on the `blas_` common-namespace
 * convention (alongside blas_char.h / blas_omp.h). `static inline` keeps them
 * codegen-neutral: at -O2 every call inlines to the bare arithmetic, exactly
 * as the former file-local statics did.
 */
#ifndef EPBLAS_PARALLEL_COMMON_BLAS_MATH_H
#define EPBLAS_PARALLEL_COMMON_BLAS_MATH_H

#include <stddef.h>

/* Round v up to the next multiple of m (m > 0). */
static inline ptrdiff_t blas_round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }

/* Smaller of two signed sizes. */
static inline ptrdiff_t blas_imin(ptrdiff_t a, ptrdiff_t b) { return a < b ? a : b; }

#endif /* EPBLAS_PARALLEL_COMMON_BLAS_MATH_H */
