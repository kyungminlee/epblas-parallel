/*
 * ytpmv — kind10 complex triangular packed matrix-vector.
 *   x := op(A)*x   (op = none / transpose / conjugate-transpose)
 *
 * Complex twin of etpmv. The serial reference is the in-place column sweep
 * (sequential); the threaded path (large N) reformulates the multiply
 * out-of-place over a private buffer and partitions columns, mirroring
 * OpenBLAS tpmv_thread:
 *   - NoTrans: column j writes a run of y (cross-column) -> per-thread private
 *     slot, controller AXPY-reduces the touched range into slot 0.
 *   - Trans/ConjTrans: y[j] = dot(op(column j), x), disjoint per thread -> all
 *     threads write disjoint y[m_from..m_to) into shared slot 0, no reduce.
 * The Trans UPPER kernel folds the diagonal (column END) in last to keep the
 * packed read stream sequential; the column kernels index a loop-invariant
 * per-column base.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef _Complex long double TC;
static inline TC cconj(TC z) { return ~z; }


#ifdef _OPENMP
/* Defined below (also used by the serial NoTrans sweep). The threaded column
 * kernel reuses it so per-thread AXPY gets the same 4x-unrolled re/im-decomposed
 * codegen that makes the serial path beat OpenBLAS. */
static void ytpmv_axpy_col(TC *restrict y_, const TC *restrict a_, TC t, ptrdiff_t m);

static inline size_t col_start_U(ptrdiff_t j) { return (size_t)j * (size_t)(j + 1) / 2; }
static inline size_t col_start_L(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)(2 * n - j + 1) / 2;
}

/* Sqrt-balanced contiguous column partition (OpenBLAS tpmv_partition, mask=7,
 * min-width 16); UPPER reversed so thread 0 takes the heaviest columns. */
static void tpmv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 7;
    const double dnum = (double)n * (double)n / (double)nthreads;
    if (!upper) {
        range[0] = 0;
        ptrdiff_t i = 0; ptrdiff_t num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask)
                    : (n - i);
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[num_cpu + 1] = range[num_cpu] + width;
            num_cpu++; i += width;
        }
        for (ptrdiff_t t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
    } else {
        range[nthreads] = n;
        ptrdiff_t i = 0; ptrdiff_t num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask)
                    : (n - i);
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width;
            num_cpu++; i += width;
        }
        for (ptrdiff_t t = 0; t < nthreads - num_cpu; ++t) range[t] = range[nthreads - num_cpu];
    }
}

static void tpmv_kernel_N(bool upper, bool nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const TC *ap, const TC *x, TC *y)
{
    if (upper) {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TC xj = x[j];
            ytpmv_axpy_col(y, &ap[cs], xj, j);            /* y[0..j) += ap[cs..]*xj */
            y[j] += nounit ? ap[cs + (size_t)j] * xj : xj;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TC xj = x[j];
            y[j] += nounit ? ap[cs] * xj : xj;
            ytpmv_axpy_col(&y[j + 1], &ap[cs + 1], xj, n - 1 - j);  /* y[j+1..n) += ap[cs+1..]*xj */
        }
    }
}

static void tpmv_kernel_T(bool upper, bool nounit, bool conj, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const TC *ap, const TC *x, TC *y)
{
#define APV(k) (conj ? cconj(ap[k]) : ap[k])
    if (upper) {
        /* Diagonal at column END (cs+j): accumulate the run first, fold diag
         * last to keep the packed read stream sequential. */
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TC s = 0.0L;
            for (ptrdiff_t i = 0; i < j; ++i) s += APV(cs + (size_t)i) * x[i];
            s += nounit ? APV(cs + (size_t)j) * x[j] : x[j];
            y[j] += s;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TC s = nounit ? APV(cs) * x[j] : x[j];
            for (ptrdiff_t i = j + 1; i < n; ++i) s += APV(cs + (size_t)(i - j)) * x[i];
            y[j] += s;
        }
    }
#undef APV
}

/* Threaded out-of-place path. Returns 1 if it handled the call, 0 to fall back
 * to the serial reference. Kept noinline so the in-place serial loops below
 * compile in a clean x87 register context. */
__attribute__((noinline))
static ptrdiff_t ytpmv_omp(bool upper, bool is_t, bool conj, bool nounit, ptrdiff_t n, ptrdiff_t incx,
                     const TC *restrict ap, TC *restrict x)
{
    ptrdiff_t nthreads = 1;
    if (n >= 50 && !omp_in_parallel()) {
        nthreads = blas_omp_max_threads();
        if (n < 500 && nthreads > 2) nthreads = 2;
    }
    if (nthreads <= 1) return 0;

    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * (ptrdiff_t)incx : 0;
    TC *buf_all = (TC *)calloc((size_t)nthreads * (size_t)n, sizeof(TC));
    ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
    TC *xbuf = NULL;
    const TC *xptr = x;
    if (incx != 1 && buf_all && range_m) {
        xbuf = (TC *)malloc((size_t)n * sizeof(TC));
        if (xbuf) {
            for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[kx + i * incx];
            xptr = xbuf;
        }
    }
    if (!(buf_all && range_m && (incx == 1 || xbuf))) {
        free(buf_all); free(range_m); if (xbuf) free(xbuf);
        return 0;
    }

    tpmv_partition(upper, n, nthreads, range_m);
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        TC *y = is_t ? buf_all : &buf_all[(size_t)tid * (size_t)n];
        ptrdiff_t m_from, m_to;
        if (upper) { m_from = range_m[nthreads - tid - 1]; m_to = range_m[nthreads - tid]; }
        else       { m_from = range_m[tid];               m_to = range_m[tid + 1]; }
        if (m_from < m_to) {
            if (is_t) tpmv_kernel_T(upper, nounit, conj, n, m_from, m_to, ap, xptr, y);
            else      tpmv_kernel_N(upper, nounit, n, m_from, m_to, ap, xptr, y);
        }
    }
    if (!is_t) {  /* reduce private slots into slot 0 over the touched range */
        if (upper) {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_to_t = range_m[nthreads - t];
                const TC *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = 0; i < m_to_t; ++i) buf_all[i] += slot[i];
            }
        } else {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_from_t = range_m[t];
                const TC *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = m_from_t; i < n; ++i) buf_all[i] += slot[i];
            }
        }
    }
    if (incx == 1) for (ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
    else           for (ptrdiff_t i = 0; i < n; ++i) x[kx + i * incx] = buf_all[i];
    free(buf_all); free(range_m); if (xbuf) free(xbuf);
    return 1;
}
#endif /* _OPENMP */

/* Forward complex AXPY  y[0..m) += t * a[0..m). t is decomposed into scalar
 * re/im long-double locals (keeps the loop-invariant multiplier off the x87
 * stack) and the body is 4x unrolled — the iterations are independent (no
 * loop-carried accumulator), so unrolling amortises loop overhead and exposes
 * ILP the way gfortran's single-wide loop cannot. Kept noinline so it compiles
 * in a clean x87 register context, isolated from the column-sweep scaffolding
 * (in-place inline here costs ~6% from register pressure). Bit-identical to the
 * _Complex form: same products, same association. */
__attribute__((noinline))
static void ytpmv_axpy_col(TC *restrict y_, const TC *restrict a_, TC t, ptrdiff_t m)
{
    const long double tr = __real__ t, ti = __imag__ t;
    long double *restrict y = (long double *)y_;
    const long double *restrict a = (const long double *)a_;
    ptrdiff_t i = 0;
    for (; i + 3 < m; i += 4) {
        for (ptrdiff_t u = 0; u < 4; ++u) {
            const long double ar = a[2 * (i + u)], ai = a[2 * (i + u) + 1];
            y[2 * (i + u)]     += tr * ar - ti * ai;
            y[2 * (i + u) + 1] += tr * ai + ti * ar;
        }
    }
    for (; i < m; ++i) {
        const long double ar = a[2 * i], ai = a[2 * i + 1];
        y[2 * i]     += tr * ar - ti * ai;
        y[2 * i + 1] += tr * ai + ti * ar;
    }
}

/* Non-conj Trans row value: dst = init + Σ_{k} ap[k]·x[k], where `init` is the
 * diagonal-scaled x[j] (gfortran's TEMP preload). The running (re,im) pair is
 * held resident on the x87 stack across the whole loop (4 loads + 2 dups, no
 * operand reloads). The pure-C single-acc form is instruction-identical to
 * gfortran but loses ~4-7% on Lower-Trans to gfortran's tighter fp80
 * accumulator-stack ordering; gcc won't reproduce it (a fxch-free C loop carved
 * into its own single-body function still measured ~1.04-1.05, A/B'd 2026-06-29),
 * and a 2nd C chain spills the 8-deep stack (trigger 6). Body transcribed from
 * the proven ytrsv U-T asm MAC, the two combines flipped subtract→add (dot, not
 * solve); ai/x as interleaved re/im. The result is stored STRAIGHT into dst
 * (= x[j]) and `init` is computed INSIDE the asm — for unit diag it is just
 * x[j] (loaded from dst); for non-unit it is x[j]·A(j,j), done as one extra MAC
 * into a zeroed accumulator. This removes BOTH the per-column store-to-local +
 * reload of init AND (non-unit) the C-side _Complex multiply, the fixed cost
 * that left small N at ~1.03 vs OpenBLAS. always_inline: the asm owns the x87
 * stack (full clobber) so it inlines into the column loop without a trigger-3
 * spill and with no per-column call. __volatile__: the result leaves only via
 * the [d] memory store, so a non-volatile asm would be elided despite "memory".
 *
 * MAC: with stack [a.im, a.re, x.im, x.re, sim, sre] (st0..st5) on entry, adds
 * Re(a·x) into sre / Im(a·x) into sim and pops the 4 operands → [sim', sre']. */
#if defined(__GNUC__) && defined(__x86_64__)
#define YTPMV_MAC                      \
        "fld %%st(1)\n\t"              \
        "fmul %%st(4),%%st\n\t"        \
        "fld %%st(1)\n\t"              \
        "fmul %%st(4),%%st\n\t"        \
        "fsubrp %%st,%%st(1)\n\t"      \
        "faddp %%st,%%st(6)\n\t"       \
        "fxch %%st(1)\n\t"             \
        "fmulp %%st,%%st(2)\n\t"       \
        "fmulp %%st,%%st(2)\n\t"       \
        "faddp %%st,%%st(1)\n\t"       \
        "faddp %%st,%%st(1)\n\t"
/* unit diagonal: init = x[j], read directly from dst (no init slot/copy) */
__attribute__((always_inline))
static inline void ytpmv_dot_u(TC *restrict dst, const TC *restrict ap_,
                               const TC *restrict x_, ptrdiff_t cnt)
{
    const long double *a = (const long double *)ap_;
    const long double *x = (const long double *)x_;
    long double *d = (long double *)dst;
    __asm__ __volatile__ (
        "fldt (%[d])\n\t"          /* sre = x[j].re */
        "fldt 16(%[d])\n\t"        /* sim = x[j].im, st1 = sre */
        "test %[i],%[i]\n\t"
        "jle 2f\n\t"
        "1:\n\t"
        "fldt (%[x])\n\t"
        "fldt 16(%[x])\n\t"
        "fldt (%[a])\n\t"
        "fldt 16(%[a])\n\t"
        YTPMV_MAC
        "add $32,%[a]\n\t"
        "add $32,%[x]\n\t"
        "sub $1,%[i]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "fstpt 16(%[d])\n\t"       /* sim -> dst.im */
        "fstpt (%[d])\n\t"         /* sre -> dst.re */
        : [a] "+r"(a), [x] "+r"(x), [i] "+r"(cnt)
        : [d] "r"(d)
        : "st", "st(1)", "st(2)", "st(3)", "st(4)",
          "st(5)", "st(6)", "st(7)", "memory", "cc");
}
/* non-unit diagonal: init = x[j]·A(j,j); the diagonal A(j,j) is at *dg_, x[j]
 * at *dst. The accumulator is zeroed and one MAC over (dg, x[j]) seeds it. */
__attribute__((always_inline))
static inline void ytpmv_dot_nu(TC *restrict dst, const TC *restrict ap_,
                                const TC *restrict x_, ptrdiff_t cnt,
                                const TC *restrict dg_)
{
    const long double *a = (const long double *)ap_;
    const long double *x = (const long double *)x_;
    const long double *g = (const long double *)dg_;
    long double *d = (long double *)dst;
    __asm__ __volatile__ (
        "fldz\n\t"                 /* sre = 0 */
        "fldz\n\t"                 /* sim = 0, st1 = sre */
        "fldt (%[d])\n\t"          /* x[j].re */
        "fldt 16(%[d])\n\t"        /* x[j].im */
        "fldt (%[g])\n\t"          /* A(j,j).re */
        "fldt 16(%[g])\n\t"        /* A(j,j).im */
        YTPMV_MAC                  /* sre,sim = Re/Im(A(j,j)·x[j]) */
        "test %[i],%[i]\n\t"
        "jle 2f\n\t"
        "1:\n\t"
        "fldt (%[x])\n\t"
        "fldt 16(%[x])\n\t"
        "fldt (%[a])\n\t"
        "fldt 16(%[a])\n\t"
        YTPMV_MAC
        "add $32,%[a]\n\t"
        "add $32,%[x]\n\t"
        "sub $1,%[i]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "fstpt 16(%[d])\n\t"       /* sim -> dst.im */
        "fstpt (%[d])\n\t"         /* sre -> dst.re */
        : [a] "+r"(a), [x] "+r"(x), [i] "+r"(cnt)
        : [d] "r"(d), [g] "r"(g)
        : "st", "st(1)", "st(2)", "st(3)", "st(4)",
          "st(5)", "st(6)", "st(7)", "memory", "cc");
}
#undef YTPMV_MAC
#else
__attribute__((noinline))
static void ytpmv_dot_u(TC *restrict dst, const TC *restrict ap_,
                        const TC *restrict x_, ptrdiff_t cnt)
{
    const long double *restrict a = (const long double *)ap_;
    const long double *restrict x = (const long double *)x_;
    long double sr = __real__ *dst, si = __imag__ *dst;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const long double ar = a[0], ai = a[1];
        const long double xr = x[0], xs = x[1];
        sr += ar * xr - ai * xs;
        si += ar * xs + ai * xr;
        a += 2; x += 2;
    }
    *dst = sr + si * 1.0iL;
}
__attribute__((noinline))
static void ytpmv_dot_nu(TC *restrict dst, const TC *restrict ap_,
                         const TC *restrict x_, ptrdiff_t cnt,
                         const TC *restrict dg_)
{
    const long double *restrict a = (const long double *)ap_;
    const long double *restrict x = (const long double *)x_;
    const TC init = (*dst) * (*dg_);
    long double sr = __real__ init, si = __imag__ init;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const long double ar = a[0], ai = a[1];
        const long double xr = x[0], xs = x[1];
        sr += ar * xr - ai * xs;
        si += ar * xs + ai * xr;
        a += 2; x += 2;
    }
    *dst = sr + si * 1.0iL;
}
#endif

/* Conjugate Trans dot: returns sum_{i<cnt} conj(ap[i]) * x[i]. Single complex
 * accumulator decomposed to scalar re/im (a 2nd chain spills the 8-deep x87
 * stack for complex — trigger 6 — and measured ~5% slower). Forward walk is
 * prefetch-friendly vs the netlib backward sweep; reassociation is within fp80
 * fuzz tol. NOINLINE so it compiles in a clean x87 context, isolated from the
 * NoTrans sweep's register allocation — inlined in-context the dot loses ~4%. */
__attribute__((noinline))
static TC ytpmv_dotc(const TC *restrict ap_, const TC *restrict x_, ptrdiff_t cnt)
{
    const long double *restrict a = (const long double *)ap_;
    const long double *restrict x = (const long double *)x_;
    long double sr = 0.0L, si = 0.0L;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const long double ar = a[0], ai = a[1];
        const long double xr = x[0], xs = x[1];
        sr += ar * xr + ai * xs;
        si += ar * xs - ai * xr;
        a += 2; x += 2;
    }
    return sr + si * 1.0iL;
}

static void ytpmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TC *restrict ap,
    TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0L + 0.0Li;
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

    /* General stride, NoTrans/ConjTrans only: gather x into contiguous
     * scratch, run the incx==1 paths, scatter back. The strided walk touches
     * x O(n^2) times at stride incx; the O(n) gather/scatter routes into the
     * contiguous fast arms (NoTrans axpy-col / ConjTrans dotc ~8-9% faster in
     * absolute terms than the references' strided walk at N=512). Plain-Trans
     * is excluded: its Lower contiguous form runs at strided-walk speed (a
     * gather would regress ~1%), the same no-headroom split as ytpsv.
     * malloc failure falls through to the direct strided walk. */
    if (incx != 1 && (TRANS == 'N' || !noconj)) {
        enum { YTPMV_STACK_N = 512 };
        TC stackbuf[YTPMV_STACK_N];
        TC *heap = NULL;
        TC *xc = (n <= YTPMV_STACK_N)
            ? stackbuf
            : (heap = (TC *)malloc((size_t)n * sizeof(TC)));
        if (xc) {
            const ptrdiff_t kx0 = (incx < 0) ? -(n - 1) * incx : 0;
            ptrdiff_t ix = kx0;
            for (ptrdiff_t i = 0; i < n; ++i) { xc[i] = x[ix]; ix += incx; }
            ytpmv_core(uplo, trans, diag, n, ap, xc, 1);
            ix = kx0;
            for (ptrdiff_t i = 0; i < n; ++i) { x[ix] = xc[i]; ix += incx; }
            free(heap);
            return;
        }
    }

#ifdef _OPENMP
    if (ytpmv_omp(UPLO == 'U', TRANS != 'N', (TRANS == 'C'), nounit, n, incx, ap, x)) return;
#endif

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TC tmp = x[j];
                        ytpmv_axpy_col(x, &ap[kk], tmp, j);  /* x[0..j) += tmp*ap[kk..) */
                        if (nounit) x[j] *= ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const TC tmp = x[j];
                        const ptrdiff_t diag = kk - (n - 1 - j);  /* ap[diag] = A(j,j) */
                        /* run x[j+1..n) forward over ap[diag+1..]; iterations are
                         * independent so forward order is bit-identical to the
                         * original descending walk and avoids the neg-stride stall */
                        ytpmv_axpy_col(&x[j + 1], &ap[diag + 1], tmp, n - 1 - j);
                        if (nounit) x[j] *= ap[diag];
                    }
                    kk -= (n - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    /* forward dot over x[0..j); ap[kk-j] pairs with x[0] */
                    if (noconj) {
                        if (nounit) ytpmv_dot_nu(&x[j], &ap[kk - j], &x[0], j, &ap[kk]);
                        else        ytpmv_dot_u(&x[j], &ap[kk - j], &x[0], j);
                    } else {
                        TC tmp = x[j];
                        if (nounit) tmp *= cconj(ap[kk]);
                        tmp += ytpmv_dotc(&ap[kk - j], &x[0], j);
                        x[j] = tmp;
                    }
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    /* forward dot over x[j+1..n); ap[kk+1] pairs with x[j+1] */
                    if (noconj) {
                        if (nounit) ytpmv_dot_nu(&x[j], &ap[kk + 1], &x[j + 1], n - 1 - j, &ap[kk]);
                        else        ytpmv_dot_u(&x[j], &ap[kk + 1], &x[j + 1], n - 1 - j);
                    } else {
                        TC tmp = x[j];
                        if (nounit) tmp *= cconj(ap[kk]);
                        tmp += ytpmv_dotc(&ap[kk + 1], &x[j + 1], n - 1 - j);
                        x[j] = tmp;
                    }
                    kk += n - j;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != zero) {
                        const TC tmp = x[jx];
                        ptrdiff_t ix = kx;
                        for (ptrdiff_t k = kk; k < kk + j; ++k) {
                            x[ix] += tmp * ap[k];
                            ix += incx;
                        }
                        if (nounit) x[jx] *= ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const TC tmp = x[jx];
                        ptrdiff_t ix = kx;
                        for (ptrdiff_t k = kk; k > kk - (n - 1 - j); --k) {
                            x[ix] += tmp * ap[k];
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= ap[kk - (n - 1 - j)];
                    }
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) tmp *= (noconj ? ap[kk] : cconj(ap[kk]));
                    for (ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                        ix -= incx;
                        tmp += (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) tmp *= (noconj ? ap[kk] : cconj(ap[kk]));
                    for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += incx;
                        tmp += (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                    kk += n - j;
                }
            }
        }
    }
}

EPBLAS_FACADE_TPMV(ytpmv, TC)
