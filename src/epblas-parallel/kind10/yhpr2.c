/*
 * yhpr2 — kind10 complex Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 */

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(N^2) complex Hermitian packed rank-2 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10): N=24 0.58/0.55, N=32 0.50/0.48,
 * N=64 0.33, N=128 0.28. N=20 marginal (0.71-0.77), so 24 is the robust floor.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YHPR2_OMP_MIN 24

typedef _Complex long double TC;
typedef long double TR;
static inline TC cconj(TC z) { return ~z; }


/* Off-diagonal rank-2 run: c[i] += xc[i]*t1 + yc[i]*t2 for i in [0,mo), all three
 * arrays contiguous and base-aligned. Both triangles reduce to this same kernel:
 * the upper column's run is over [0,j) at the array bases; the lower column's run
 * is over [0,N-1-j) at x+j+1 / y+j+1 / c0+1. Inputs are unit-stride here
 * (native-contiguous input or the gathered scratch).
 *
 * Hand-written x87 inline asm transcribing gfortran's strided `.L40` schedule —
 * the fastest code any compiler emits for this kernel (43.6k vs gcc's 46.8k,
 * equal 9 fldt/elt; the gap is pure x87 instruction scheduling that gcc will not
 * reproduce from C, see task 16 / project_l2_strided_gather). The 4 loop-invariant
 * constants stay resident (stack top->bottom t1i,t2r,t1r,t2i); the pointer
 * increments are interleaved exactly as gfortran places them (AP early after the
 * first load with -16/-32 back-offsets, X mid-loop, Y late). Same left-folded
 * association ((AP + X*T1) + Y*T2) the references use — bit-exact (fuzz relerr 0).
 * incx==incy==1 here, so all increments are +32 bytes. no_stack_protector: the
 * "m" constant operands + "memory" clobber trip -fstack-protector-strong, adding
 * a per-column canary prologue gfortran's inlined .L40 has not; the asm touches
 * no stack buffer so the guard is pure overhead. always_inline folds the run into
 * the col helper so each column is a single call (contig->col), matching the
 * references' fully-inlined column body instead of layering contig->col->run. */
__attribute__((always_inline, no_stack_protector))
static inline void yhpr2_run(ptrdiff_t mo, TC t1, TC t2,
                      const TC *restrict xc, const TC *restrict yc, TC *restrict c) {
    if (mo <= 0) return;
    /* Alias the by-value complex params as {re,im} pairs so the asm loads the
     * constants straight from their incoming stack slots — referencing fresh
     * `__real__ t1` locals instead made gcc reload-then-restore each into a new
     * slot (12 fp80 mem ops/column vs 4). */
    const TR *p1 = (const TR *)&t1, *p2 = (const TR *)&t2;
    const TC *end = c + mo;          /* post-increment sentinel */
    __asm__ volatile(
        "fldt %[t2i]\n\t"            /* st: t2i */
        "fldt %[t1r]\n\t"           /* st: t1r t2i */
        "fldt %[t2r]\n\t"           /* st: t2r t1r t2i */
        "fldt %[t1i]\n\t"           /* st: t1i t2r t1r t2i  (resident) */
        ".p2align 4\n\t"            /* match gfortran .L40 (16B); 32B was no better */
        ".p2align 3\n\t"
        "1:\n\t"
        "fldt (%[x])\n\t"           /* Xr */
        "addq $32, %[c]\n\t"        /* AP += 32 (early) */
        "fldt 16(%[x])\n\t"         /* Xi */
        "fmul %%st(4), %%st\n\t"
        "fld %%st(2)\n\t"
        "fmul %%st(2), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"   /* Im(X*T1) */
        "fldt -16(%[c])\n\t"        /* AP.im */
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"           /* Yr */
        "fmul %%st(6), %%st\n\t"
        "fldt 16(%[y])\n\t"         /* Yi */
        "fmul %%st(5), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"   /* Im(Y*T2) */
        "faddp %%st, %%st(1)\n\t"   /* AP.im result */
        "fxch %%st(1)\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[x])\n\t"         /* Xi reload */
        "addq $32, %[x]\n\t"        /* X += 32 (mid) */
        "fmul %%st(3), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"  /* Re(X*T1) */
        "fldt -32(%[c])\n\t"        /* AP.re */
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"           /* Yr reload */
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[y])\n\t"         /* Yi reload */
        "addq $32, %[y]\n\t"        /* Y += 32 (late) */
        "fmul %%st(7), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"  /* Re(Y*T2) */
        "faddp %%st, %%st(1)\n\t"   /* AP.re result */
        "fstpt -32(%[c])\n\t"
        "fstpt -16(%[c])\n\t"
        "cmpq %[end], %[c]\n\t"
        "jne 1b\n\t"
        "fstp %%st(0)\n\t"          /* drop the 4 resident constants */
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        : [x] "+r"(xc), [y] "+r"(yc), [c] "+r"(c)
        : [end] "r"(end),
          [t1r] "m"(p1[0]), [t1i] "m"(p1[1]), [t2r] "m"(p2[0]), [t2i] "m"(p2[1])
        : "memory", "cc",
          "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)");
}

/* Strided twin of yhpr2_run: identical .L40 schedule, but x and y advance by the
 * caller's byte strides (sx = incx*sizeof(TC), sy = incy*sizeof(TC)) instead of a
 * fixed +32. c (the packed column) is always contiguous, so its +32 advance and
 * the cmpq sentinel are unchanged. This lets the SERIAL strided path walk the
 * input vectors in place — exactly matching gfortran's strided .L40 — instead of
 * paying the O(N) gather the threaded path uses. addq reg,ptr is the same cost as
 * addq imm,ptr, and sx/sy hoist out of the loop, so this is schedule-identical to
 * the contiguous core for the work that matters. Kept as a separate body so the
 * proven contiguous core stays byte-for-byte unperturbed. */
__attribute__((always_inline, no_stack_protector))
static inline void yhpr2_run_strided(ptrdiff_t mo, TC t1, TC t2,
                      const TC *restrict xc, const TC *restrict yc, TC *restrict c,
                      ptrdiff_t sx, ptrdiff_t sy) {
    if (mo <= 0) return;
    const TR *p1 = (const TR *)&t1, *p2 = (const TR *)&t2;
    const TC *end = c + mo;
    __asm__ volatile(
        "fldt %[t2i]\n\t"
        "fldt %[t1r]\n\t"
        "fldt %[t2r]\n\t"
        "fldt %[t1i]\n\t"
        ".p2align 4\n\t"
        ".p2align 3\n\t"
        "1:\n\t"
        "fldt (%[x])\n\t"
        "addq $32, %[c]\n\t"
        "fldt 16(%[x])\n\t"
        "fmul %%st(4), %%st\n\t"
        "fld %%st(2)\n\t"
        "fmul %%st(2), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt -16(%[c])\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"
        "fmul %%st(6), %%st\n\t"
        "fldt 16(%[y])\n\t"
        "fmul %%st(5), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fxch %%st(1)\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[x])\n\t"
        "addq %[sx], %[x]\n\t"      /* X += incx*sizeof(TC) */
        "fmul %%st(3), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"
        "fldt -32(%[c])\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[y])\n\t"
        "addq %[sy], %[y]\n\t"      /* Y += incy*sizeof(TC) */
        "fmul %%st(7), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fstpt -32(%[c])\n\t"
        "fstpt -16(%[c])\n\t"
        "cmpq %[end], %[c]\n\t"
        "jne 1b\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        : [x] "+r"(xc), [y] "+r"(yc), [c] "+r"(c)
        : [end] "r"(end), [sx] "r"(sx), [sy] "r"(sy),
          [t1r] "m"(p1[0]), [t1i] "m"(p1[1]), [t2r] "m"(p2[0]), [t2i] "m"(p2[1])
        : "memory", "cc",
          "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)");
}

/* Per-column rank-2 updates: off-diagonal run via the hand-tuned asm core, then
 * the Hermitian diagonal forced real (the run last would dirty the x87 stack, so
 * the single diagonal write follows). Upper run is [0,j) at the array bases;
 * lower run is [0,N-1-j) at the j+1 offsets. */
__attribute__((always_inline))
static inline void yhpr2_col_upper(ptrdiff_t j, TC t1, TC t2,
                            const TC *restrict x, const TC *restrict y, TC *restrict ap) {
    TC *restrict c = ap + (size_t)j * (j + 1) / 2;
    yhpr2_run(j, t1, t2, x, y, c);
    c[j] = (TR)__real__ c[j] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

__attribute__((always_inline))
static inline void yhpr2_col_lower(ptrdiff_t j, ptrdiff_t n, TC t1, TC t2,
                            const TC *restrict x, const TC *restrict y, TC *restrict ap) {
    const ptrdiff_t mo = n - j - 1;
    TC *restrict c0 = ap + ((size_t)j * n - (size_t)j * (j - 1) / 2);
    yhpr2_run(mo, t1, t2, x + j + 1, y + j + 1, c0 + 1);
    c0[0] = (TR)__real__ c0[0] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

/* Strided column twins: x/y are walked with incx/incy via yhpr2_run_strided (no
 * gather). kx/ky are the negative-stride base offsets; jx/jy index the diagonal
 * vector element. Same packed-column layout and accumulation order as the unit
 * helpers, so bit-identical to the gathered path. */
__attribute__((always_inline))
static inline void yhpr2_col_upper_s(ptrdiff_t j, TC t1, TC t2,
                            const TC *restrict x, ptrdiff_t incx, ptrdiff_t kx,
                            const TC *restrict y, ptrdiff_t incy, ptrdiff_t ky,
                            TC *restrict ap) {
    TC *restrict c = ap + (size_t)j * (j + 1) / 2;
    const ptrdiff_t es = (ptrdiff_t)sizeof(TC);
    yhpr2_run_strided(j, t1, t2, x + kx, y + ky, c, incx * es, incy * es);
    const ptrdiff_t jx = kx + j * incx, jy = ky + j * incy;
    c[j] = (TR)__real__ c[j] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
}

__attribute__((always_inline))
static inline void yhpr2_col_lower_s(ptrdiff_t j, ptrdiff_t n, TC t1, TC t2,
                            const TC *restrict x, ptrdiff_t incx, ptrdiff_t kx,
                            const TC *restrict y, ptrdiff_t incy, ptrdiff_t ky,
                            TC *restrict ap) {
    const ptrdiff_t mo = n - j - 1;
    TC *restrict c0 = ap + ((size_t)j * n - (size_t)j * (j - 1) / 2);
    const ptrdiff_t es = (ptrdiff_t)sizeof(TC);
    const ptrdiff_t jx = kx + j * incx, jy = ky + j * incy;
    yhpr2_run_strided(mo, t1, t2, x + jx + incx, y + jy + incy, c0 + 1,
                      incx * es, incy * es);
    c0[0] = (TR)__real__ c0[0] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
}

/* Serial strided dispatch: walk columns in place via the strided asm core. Used
 * only when the run would NOT thread; the threaded strided path still gathers
 * into the contiguous core below (so the omp scaling is preserved). */
static void yhpr2_strided(char UPLO, ptrdiff_t n, TC alpha,
                          const TC *restrict x, ptrdiff_t incx,
                          const TC *restrict y, ptrdiff_t incy, TC *restrict ap)
{
    const TC zero = 0.0L + 0.0Li;
    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
    const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
    ptrdiff_t jx = kx, jy = ky;
    if (UPLO == 'U') {
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[jx] != zero || y[jy] != zero)
                yhpr2_col_upper_s(j, alpha * cconj(y[jy]), cconj(alpha * x[jx]),
                                  x, incx, kx, y, incy, ky, ap);
            else {
                const size_t kk = (size_t)j * (j + 1) / 2;
                ap[kk + j] = (TR)__real__ ap[kk + j];
            }
            jx += incx; jy += incy;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[jx] != zero || y[jy] != zero)
                yhpr2_col_lower_s(j, n, alpha * cconj(y[jy]), cconj(alpha * x[jx]),
                                  x, incx, kx, y, incy, ky, ap);
            else {
                const size_t kk = (size_t)j * n - (size_t)j * (j - 1) / 2;
                ap[kk] = (TR)__real__ ap[kk];
            }
            jx += incx; jy += incy;
        }
    }
}

/* Unit-stride dispatch, shared by the contiguous fast path and the gathered
 * strided path. schedule(static,1): column j touches j (upper) or N-1-j (lower)
 * off-diagonal packed elements, so a contiguous static block hands one thread
 * the heavy triangle end and starves the rest (par caps at ~2x on 4 cores).
 * Cyclic static,1 interleaves short and long columns across the team, balancing
 * the skew symmetrically for both UPLO. The Hermitian diagonal is forced real
 * every column — including the skipped (x[j]==y[j]==0) ones — so the else branch
 * still writes it. */
static void yhpr2_contig(char UPLO, ptrdiff_t n, TC alpha,
                         const TC *restrict x, const TC *restrict y, TC *restrict ap)
{
    const TC zero = 0.0L + 0.0Li;
    if (UPLO == 'U') {
#ifdef _OPENMP
        const bool use_omp = (n >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero || y[j] != zero)
                yhpr2_col_upper(j, alpha * cconj(y[j]), cconj(alpha * x[j]), x, y, ap);
            else {
                const size_t kk = (size_t)j * (j + 1) / 2;
                ap[kk + j] = (TR)__real__ ap[kk + j];
            }
        }
    } else {
#ifdef _OPENMP
        const bool use_omp = (n >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero || y[j] != zero)
                yhpr2_col_lower(j, n, alpha * cconj(y[j]), cconj(alpha * x[j]), x, y, ap);
            else {
                const size_t kk = (size_t)j * n - (size_t)j * (j - 1) / 2;
                ap[kk] = (TR)__real__ ap[kk];
            }
        }
    }
}

static void yhpr2_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict ap)
{
    const TC alpha = *alpha_;
    const TC zero = 0.0L + 0.0Li;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        yhpr2_contig(UPLO, n, alpha, x, y, ap);
        return;
    }

    /* Serial strided: walk the inputs in place via the strided asm core (no
     * gather), exactly matching gfortran's strided .L40. Only the threaded path
     * below gathers — there the O(N) gather is dwarfed by the O(N^2) threaded
     * work and buys the contiguous core's omp scaling; at serial small N the
     * gather is the whole ~2% gap to gfortran, so we skip it. The predicate
     * mirrors yhpr2_contig's internal use_omp so a would-thread run is never
     * sent down the serial path. */
#ifdef _OPENMP
    const bool would_thread = (n >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
#else
    const bool would_thread = false;
#endif
    if (!would_thread) {
        yhpr2_strided(UPLO, n, alpha, x, incx, y, incy, ap);
        return;
    }

    /* General stride: gather x and y into contiguous scratch and run the
     * stride-1 core — which already beats both refs and threads. x and y are
     * read-only here (only ap is written), so unlike esymv there is no
     * scatter-back. The gather is O(N) against O(N^2) work, so it is free past
     * tiny N; the strided per-element walk below loses ~4-9% to the references
     * (placement-bound, see project_l2_strided_gather). Same column-order
     * accumulation as the direct walk, so bit-identical. Falls back to the
     * direct strided loop if the scratch allocation fails. */
    {
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        TC stackbuf[2 * 256];
        TC *heap = NULL;
        TC *xc, *yc;
        if (n <= 256) {
            xc = stackbuf; yc = stackbuf + n;
        } else {
            heap = (TC *)malloc((size_t)2 * n * sizeof(TC));
            xc = heap; yc = heap ? heap + n : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t k = 0; k < n; ++k) {
                xc[k] = x[ix]; yc[k] = y[iy];
                ix += incx; iy += incy;
            }
            yhpr2_contig(UPLO, n, alpha, xc, yc, ap);
            free(heap);
            return;
        }
        free(heap);
    }

    {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TC t1 = alpha * cconj(y[jy]);
                    const TC t2 = cconj(alpha * x[jx]);
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                    ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
                } else {
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            /* Direct strided lower walk — now only the malloc-failure fallback;
             * the common strided path gathers into the contiguous core above. */
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TC t1 = alpha * cconj(y[jy]);
                    const TC t2 = cconj(alpha * x[jx]);
                    ptrdiff_t ix = jx, iy = jy;
                    ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
                    for (ptrdiff_t k = kk + 1; k < kk + (n - j); ++k) {
                        ix += incx; iy += incy;
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                    }
                } else {
                    ap[kk] = (TR)__real__ ap[kk];
                }
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(yhpr2, TC)
