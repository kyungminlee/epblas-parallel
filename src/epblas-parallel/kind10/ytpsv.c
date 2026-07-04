/*
 * ytpsv — kind10 complex triangular packed solve.
 *   x := inv(A)*x, inv(A^T)*x, or inv(A^H)*x
 *
 * Contiguous (incx==1) serial arms use the twins' proven fp80 machinery:
 * NoTrans column updates go through a noinline 4x-unrolled re/im-decomposed
 * subtract-AXPY (ytpmv_axpy_col's solve form — same products, same
 * association, iterations independent, so the Upper arm's backward netlib
 * walk is flipped forward bit-identically); ConjTrans row values come from a
 * noinline single-accumulator re/im-decomposed solve dot seeded with x[j]
 * (ytpmv_dotc's solve form) with the conjugation folded into the accumulate
 * signs (no fchs on the critical path). Upper-ConjTrans keeps netlib's
 * ascending order (bit-identical); Lower-ConjTrans walks the packed column
 * FORWARD like ytrsv L-T (netlib is descending; x fits L1 at these sizes —
 * reassociation within fp80 fuzz tol). Plain-Trans keeps ytpsv_serial's
 * verbatim netlib loop — asm-MAC and decomposed-dot variants both measured
 * slower there (no fchs to remove, no spill to fix). Strided calls stay on
 * verbatim netlib.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define YTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define YTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define YTPSV_MAX_CPUS 256
#endif

typedef _Complex long double TC;
static inline TC cconj(TC z) { return ~z; }


/* Matrix element with optional conjugation ('C' ⇒ conjugate, 'T' ⇒ as-is). */
static inline TC yelem(TC a, bool noconj) {
    return noconj ? a : cconj(a);
}

#ifdef _OPENMP
/* Column base offsets into the packed array (column-major triangle).
 *   Lower: column j starts at its diagonal (row j); element (i,j) i>=j at base+(i-j).
 *   Upper: column j starts at row 0;            element (i,j) i<=j at base+i.       */
static inline size_t cbL(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)n - (size_t)j * (size_t)(j - 1) / 2;
}
static inline size_t cbU(ptrdiff_t j) {
    return (size_t)j * (size_t)(j + 1) / 2;
}
#endif

/* Bit-exact serial path (verbatim reference). Live only for incx!=1 — the
 * incx==1 arms are superseded by ytpsv_contig below (kept verbatim as the
 * reference transcription). noconj = (TRANS=='T'); NoTrans never conjugates. */
static void ytpsv_serial(char UPLO, char TRANS, bool noconj, bool nounit,
                         ptrdiff_t n, const TC *restrict ap, TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0L + 0.0Li;
    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const TC tmp = x[j];
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { x[i] -= tmp * ap[k]; --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const TC tmp = x[j];
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < n; ++i) { x[i] -= tmp * ap[k]; ++k; }
                    }
                    kk += n - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = 0; i < j; ++i) { tmp -= ap[k] * x[i]; ++k; }
                    else        for (ptrdiff_t i = 0; i < j; ++i) { tmp -= cconj(ap[k]) * x[i]; ++k; }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = n - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    else        for (ptrdiff_t i = n - 1; i > j; --i) { tmp -= cconj(ap[k]) * x[i]; --k; }
                    if (nounit) tmp /= (noconj ? ap[kk - (n - 1 - j)] : cconj(ap[kk - (n - 1 - j)]));
                    x[j] = tmp;
                    kk -= (n - j);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const TC tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const TC tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                            ix += incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx += incx;
                    kk += n - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
                    x[jx] = tmp;
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k > kk - (n - 1 - j); --k) {
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk - (n - 1 - j)] : cconj(ap[kk - (n - 1 - j)]));
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    }
}

/* ---- contiguous (incx==1) serial machinery — see header comment ---- */

/* Forward complex subtract-AXPY  y[0..m) -= t * a[0..m)  (the solve twin of
 * ytpmv_axpy_col). t is decomposed into scalar re/im long-double locals (keeps
 * the loop-invariant multiplier off the x87 stack) and the body is 4x unrolled
 * — iterations are independent (no loop-carried accumulator), so unrolling
 * amortises loop overhead. Kept noinline so it compiles in a clean x87
 * register context, isolated from the column-sweep scaffolding. Bit-identical
 * to the _Complex form: same products, same association. */
__attribute__((noinline))
static void ytpsv_axpy_col(TC *restrict y_, const TC *restrict a_, TC t, ptrdiff_t m)
{
    const long double tr = __real__ t, ti = __imag__ t;
    long double *restrict y = (long double *)y_;
    const long double *restrict a = (const long double *)a_;
    ptrdiff_t i = 0;
    for (; i + 3 < m; i += 4) {
        for (int u = 0; u < 4; ++u) {
            const long double ar = a[2 * (i + u)], ai = a[2 * (i + u) + 1];
            y[2 * (i + u)]     -= tr * ar - ti * ai;
            y[2 * (i + u) + 1] -= tr * ai + ti * ar;
        }
    }
    for (; i < m; ++i) {
        const long double ar = a[2 * i], ai = a[2 * i + 1];
        y[2 * i]     -= tr * ar - ti * ai;
        y[2 * i + 1] -= tr * ai + ti * ar;
    }
}

/* Conj Trans row value: returns *xi − Σ_{k=0}^{m-1} cconj(ai[k])·x[k], forward
 * walk, conjugation folded into the accumulate signs (no fchs on the critical
 * path — the fchs is what disrupts gcc's scheduling on the conj arms; cf.
 * ytpmv_dotc). Single decomposed accumulator (a 2nd chain spills the 8-deep
 * stack, x87 trigger 6). Bit-identical to the _Complex single-running-sum form
 * over the same element order. */
__attribute__((noinline))
static TC ytpsv_uc_solve(const TC *restrict ai_, const TC *restrict x_,
                         const TC *restrict xi, ptrdiff_t m)
{
    const long double *restrict a = (const long double *)ai_;
    const long double *restrict x = (const long double *)x_;
    long double sr = __real__ *xi, si = __imag__ *xi;
    for (ptrdiff_t k = 0; k < m; ++k) {
        const long double ar = a[2 * k], ac = a[2 * k + 1];
        const long double xr = x[2 * k], xs = x[2 * k + 1];
        sr -= ar * xr + ac * xs;
        si -= ar * xs - ac * xr;
    }
    return sr + si * 1.0iL;
}

/* Contiguous serial solve for the NoTrans and ConjTrans arms. Same recurrence
 * as ytpsv_serial's incx==1 half with the inner loops routed through the
 * helpers above. NoTrans arms and Upper-ConjTrans are bit-identical to netlib;
 * Lower-ConjTrans walks the packed column forward (netlib descends — ytrsv L-T
 * precedent, reassociation within fp80 fuzz tol). Plain-Trans (noconj) is NOT
 * routed here: its verbatim netlib loop in ytpsv_serial is already optimal —
 * both the ytrsv_ut_solve x87 asm MAC (~1-2% worse) and this file's decomposed
 * dot shape (~1-4% worse) were A/B'd and lost to gcc's schedule of the plain
 * _Complex loop (no fchs to eliminate, no scaffolding spill to fix). */
static void ytpsv_contig(char UPLO, char TRANS, bool noconj, bool nounit,
                         ptrdiff_t n, const TC *restrict ap, TC *restrict x)
{
    const TC zero = 0.0L + 0.0Li;
    (void)noconj;
    if (TRANS == 'N') {
        if (UPLO == 'U') {
            ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
            for (ptrdiff_t j = n - 1; j >= 0; --j) {
                if (x[j] != zero) {
                    if (nounit) x[j] /= ap[kk];
                    /* col j off-diag: x[i] ↔ ap[kk-j+i], i=0..j-1 (forward) */
                    ytpsv_axpy_col(x, &ap[kk - j], x[j], j);
                }
                kk -= j + 1;
            }
        } else {
            ptrdiff_t kk = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != zero) {
                    if (nounit) x[j] /= ap[kk];
                    ytpsv_axpy_col(&x[j + 1], &ap[kk + 1], x[j], n - 1 - j);
                }
                kk += n - j;
            }
        }
    } else {
        if (UPLO == 'U') {
            ptrdiff_t kk = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                TC tmp = ytpsv_uc_solve(&ap[kk], x, &x[j], j);
                if (nounit) tmp /= cconj(ap[kk + j]);
                x[j] = tmp;
                kk += j + 1;
            }
        } else {
            ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
            for (ptrdiff_t j = n - 1; j >= 0; --j) {
                /* diag at dg = kk-(n-1-j); x[i] ↔ ap[dg+i-j], i=j+1..n-1 */
                const ptrdiff_t dg = kk - (n - 1 - j);
                TC tmp = ytpsv_uc_solve(&ap[dg + 1], &x[j + 1], &x[j], n - 1 - j);
                if (nounit) tmp /= cconj(ap[dg]);
                x[j] = tmp;
                kk -= (n - j);
            }
        }
    }
}

#ifdef _OPENMP
/* Solve a single diagonal block [j0,j1) in packed storage (within-block coupling
 * only). Threaded path need only match serial within fp80 fuzz tol. */
static void ytpsv_block(char UPLO, char TRANS, bool noconj, bool nounit,
                        ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t n,
                        const TC *restrict ap, TC *restrict x)
{
    const TC zero = 0.0L + 0.0Li;
    const bool lower = (UPLO == 'L');
    if (TRANS == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                if (x[j] == zero) continue;
                const size_t b = cbU(j);
                if (nounit) x[j] /= ap[b + j];
                const TC tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) x[i] -= tmp * ap[b + i];
            }
        } else {                                        /* Lower: forward */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                if (x[j] == zero) continue;
                const size_t b = cbL(j, n);
                if (nounit) x[j] /= ap[b];
                const TC tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) x[i] -= tmp * ap[b + (i - j)];
            }
        }
    } else {
        if (!lower) {                                   /* Upper^T/H: forward, k<j */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                const size_t b = cbU(j);
                TC tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) tmp -= yelem(ap[b + i], noconj) * x[i];
                if (nounit) tmp /= yelem(ap[b + j], noconj);
                x[j] = tmp;
            }
        } else {                                        /* Lower^T/H: backward, k>j */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                const size_t b = cbL(j, n);
                TC tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) tmp -= yelem(ap[b + (i - j)], noconj) * x[i];
                if (nounit) tmp /= yelem(ap[b], noconj);
                x[j] = tmp;
            }
        }
    }
}

/* Blocked threaded packed solve, incx==1 only. Loop-carried dependence confined
 * to small YTPSV_BLK diagonal blocks (solved serially); the bulk O(N^2)
 * off-diagonal coupling is threaded over disjoint output rows. Returns 1 if it
 * handled the call. */
__attribute__((noinline)) static bool ytpsv_omp(
    char UPLO, char TRANS, bool noconj, bool nounit, ptrdiff_t n,
    const TC *restrict ap, TC *restrict x)
{
    if (n < YTPSV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > YTPSV_MAX_CPUS) nthreads = YTPSV_MAX_CPUS;
    const TC zero = 0.0L + 0.0Li;
    const bool lower = (UPLO == 'L');
    const bool trans = (TRANS != 'N');

    if (!trans) {
        if (lower) {
            for (ptrdiff_t j0 = 0; j0 < n; j0 += YTPSV_BLK) {
                ptrdiff_t j1 = j0 + YTPSV_BLK; if (j1 > n) j1 = n;
                ytpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
                if (j1 >= n) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    ptrdiff_t tid = omp_get_thread_num();
                    ptrdiff_t rlo = j1 + blas_part_bound((n - j1), tid, nthreads);
                    ptrdiff_t rhi = j1 + blas_part_bound((n - j1), tid + 1, nthreads);
                    for (ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (xi == zero) continue;
                        const TC *restrict col = &ap[cbL(i, n)];     /* col[k-i] = A(k,i) */
                        for (ptrdiff_t k = rlo; k < rhi; ++k) x[k] -= xi * col[k - i];
                    }
                }
            }
        } else {
            for (ptrdiff_t j1 = n; j1 > 0; j1 -= YTPSV_BLK) {
                ptrdiff_t j0 = j1 - YTPSV_BLK; if (j0 < 0) j0 = 0;
                ytpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    ptrdiff_t tid = omp_get_thread_num();
                    ptrdiff_t rlo = blas_part_bound(j0, tid, nthreads);
                    ptrdiff_t rhi = blas_part_bound(j0, tid + 1, nthreads);
                    for (ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (xi == zero) continue;
                        const TC *restrict col = &ap[cbU(i)];        /* col[k] = A(k,i) */
                        for (ptrdiff_t k = rlo; k < rhi; ++k) x[k] -= xi * col[k];
                    }
                }
            }
        }
    } else {
        if (lower) {                                       /* backward, k > j */
            for (ptrdiff_t j1 = n; j1 > 0; j1 -= YTPSV_BLK) {
                ptrdiff_t j0 = j1 - YTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < n) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        ptrdiff_t tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound((j1 - j0), tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound((j1 - j0), tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *restrict col = &ap[cbL(i, n)];
                            TC s = zero;
                            for (ptrdiff_t k = j1; k < n; ++k) s += yelem(col[k - i], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                ytpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (ptrdiff_t j0 = 0; j0 < n; j0 += YTPSV_BLK) {
                ptrdiff_t j1 = j0 + YTPSV_BLK; if (j1 > n) j1 = n;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        ptrdiff_t tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound((j1 - j0), tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound((j1 - j0), tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *restrict col = &ap[cbU(i)];
                            TC s = zero;
                            for (ptrdiff_t k = 0; k < j0; ++k) s += yelem(col[k], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                ytpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
            }
        }
    }
    return 1;
}
#endif

static void ytpsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TC *restrict ap,
    TC *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (incx == 1 && n >= YTPSV_OMP_MIN && blas_omp_max_threads() > 1
        && ytpsv_omp(UPLO, TRANS, noconj, nounit, n, ap, x))
        return;
#endif

    /* Contiguous NoTrans/ConjTrans take the helper-routed arms; contiguous
     * plain-Trans stays on ytpsv_serial's verbatim netlib loop (fastest form,
     * see ytpsv_contig comment), as does every strided call. */
    if (incx == 1 && (TRANS == 'N' || !noconj)) {
        ytpsv_contig(UPLO, TRANS, noconj, nounit, n, ap, x);
        return;
    }
    ytpsv_serial(UPLO, TRANS, noconj, nounit, n, ap, x, incx);
}

EPBLAS_FACADE_TPMV(ytpsv, TC)
