/*
 * etpmv — kind10 (long double) triangular packed matrix-vector.
 *   x := A*x or A^T*x
 *
 * The serial reference is the in-place column sweep (each column updates a run
 * of x in dependency order), which is inherently sequential. But the operation
 * is a pure multiply — every *output* element is an independent dot/axpy — so
 * the threaded path (large N) reformulates it out-of-place over a private
 * buffer and partitions columns across threads, mirroring OpenBLAS tpmv_thread:
 *   - NoTrans: column j writes a run of y (cross-column) -> per-thread private
 *     slot, controller AXPY-reduces the touched range into slot 0.
 *   - Trans:   y[j] is a dot of column j with x (disjoint per thread) -> all
 *     threads write disjoint y[m_from..m_to) into the shared slot 0, no reduce.
 * Final copy writes the buffer back to x. The column kernels index a
 * loop-invariant per-column base (ap[cs + i]) to keep the fp80 loop body off
 * a live running packed index.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef long double TR;

/* Two-chain forward dot for the Trans serial paths: y[j] = dot(column, x). Two
 * fp80 accumulators hide the ~5-cyc fadd latency (no SIMD/FMA on x87 — chain
 * count is the only lever) and the forward walk is prefetch-friendly vs the
 * netlib backward sweep. xinc lets the strided path reuse the same kernel.
 * Reassociation within fp80 fuzz tol (kind10-serial=netlib rule removed
 * 2026-06-10). NOINLINE so it does not perturb the x87 register allocation of
 * the in-place NoTrans serial sweep in etpmv_core (inlining it regressed
 * Lower-NoTrans large-N ~12%); the per-column call is amortised over the O(N)
 * dot, same as etpmv_omp being carved out for the same reason. */
__attribute__((noinline))
static TR etpmv_dot2(const TR *restrict ap, const TR *restrict x,
                     ptrdiff_t cnt, ptrdiff_t xinc) {
    TR t0 = 0.0L, t1 = 0.0L;
    ptrdiff_t i = 0, ix = 0;
    for (; i + 1 < cnt; i += 2) {
        t0 += ap[i]     * x[ix];
        t1 += ap[i + 1] * x[ix + xinc];
        ix += 2 * xinc;
    }
    if (i < cnt) t0 += ap[i] * x[ix];
    return t0 + t1;
}


/* NoTrans serial handler — a byte-faithful copy of the OpenBLAS reference serial
 * body (both NoTrans AND Trans arms present). Only ever invoked for NoTrans, but
 * the Trans arm is retained VERBATIM because the whole-function instruction
 * layout is what lands the hot UNN/LNN axpy loops in the CPU's DSB (uop-cache)
 * rather than the legacy decoders (MITE) on Coffee Lake — any deviation (delete
 * the Trans arm, swap in a 2-chain, or a helper call) reverts UNN to MITE and
 * costs ~11% at small N. noinline+noclone pin the standalone codegen and block
 * IPA from specialising away the dead Trans arm (verified DSB-resident with the
 * literal-'N' call site under -O3 -march=native, gcc-15). See task notes. */
__attribute__((noinline, noclone))
static void epblas_etpmv_notrans_dsb(char uplo_c, char trans_c, char diag_c,
                                     ptrdiff_t n, const TR *ap, TR *x, ptrdiff_t incx)
{
    bool upper  = (blas_up(uplo_c) == 'U');
    char trc    = blas_up(trans_c);
    bool trans  = (trc == 'T' || trc == 'C');
    bool nounit = (blas_up(diag_c) == 'N');

    if (n == 0) return;
    if (incx < 0) x -= (n - 1) * incx;
    ptrdiff_t kx = 0;

    if (!trans) {
        if (upper) {
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0L) {
                        TR temp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = 0; i < j; ++i) { x[i] += temp * ap[k]; ++k; }
                        if (nounit) x[j] *= ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0L) {
                        TR temp = x[jx];
                        ptrdiff_t ix = kx;
                        for (ptrdiff_t k = kk; k < kk + j; ++k) {
                            x[ix] += temp * ap[k];
                            ix += incx;
                        }
                        if (nounit) x[jx] *= ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            }
        } else {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0L) {
                        TR temp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[i] += temp * ap[k]; --k; }
                        if (nounit) x[j] *= ap[kk - (n - 1 - j)];
                    }
                    kk -= (n - j);
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0L) {
                        TR temp = x[jx];
                        ptrdiff_t ix = kx;
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = n - 1; i > j; --i) {
                            x[ix] += temp * ap[k];
                            ix -= incx;
                            --k;
                        }
                        if (nounit) x[jx] *= ap[kk - (n - 1 - j)];
                    }
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    } else {
        if (upper) {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR temp = x[j];
                    if (nounit) temp *= ap[kk];
                    ptrdiff_t k = kk - 1;
                    for (ptrdiff_t i = j - 1; i >= 0; --i) { temp += ap[k] * x[i]; --k; }
                    x[j] = temp;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR temp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) temp *= ap[kk];
                    ptrdiff_t k = kk - 1;
                    for (ptrdiff_t i = j - 1; i >= 0; --i) {
                        ix -= incx;
                        temp += ap[k] * x[ix];
                        --k;
                    }
                    x[jx] = temp;
                    jx -= incx;
                    kk -= j + 1;
                }
            }
        } else {
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = x[j];
                    if (nounit) temp *= ap[kk];
                    ptrdiff_t k = kk + 1;
                    for (ptrdiff_t i = j + 1; i < n; ++i) { temp += ap[k] * x[i]; ++k; }
                    x[j] = temp;
                    kk += (n - j);
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) temp *= ap[kk];
                    ptrdiff_t k = kk + 1;
                    for (ptrdiff_t i = j + 1; i < n; ++i) {
                        ix += incx;
                        temp += ap[k] * x[ix];
                        ++k;
                    }
                    x[jx] = temp;
                    jx += incx;
                    kk += (n - j);
                }
            }
        }
    }
}


#ifdef _OPENMP
static inline size_t col_start_U(ptrdiff_t j) { return (size_t)j * (size_t)(j + 1) / 2; }
static inline size_t col_start_L(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)(2 * n - j + 1) / 2;
}

/* Sqrt-balanced contiguous column partition (OpenBLAS tpmv_partition, mask=7,
 * min-width 16). UPPER reverses the assignment so thread 0 takes the highest
 * (heaviest) columns; LOWER is forward. */
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
                          const TR *ap, const TR *x, TR *y)
{
    if (upper) {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TR xj = x[j];
            for (ptrdiff_t i = 0; i < j; ++i) y[i] += ap[cs + (size_t)i] * xj;
            y[j] += nounit ? ap[cs + (size_t)j] * xj : xj;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TR xj = x[j];
            y[j] += nounit ? ap[cs] * xj : xj;
            for (ptrdiff_t i = j + 1; i < n; ++i) y[i] += ap[cs + (size_t)(i - j)] * xj;
        }
    }
}

static void tpmv_kernel_T(bool upper, bool nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const TR *ap, const TR *x, TR *y)
{
    if (upper) {
        /* Diagonal (ap[cs+j]) sits at the END of the column, so accumulate the
         * cs+0..cs+j-1 run first and fold the diagonal in last — keeping the
         * packed read stream sequential (diag-first jumped back ~5-8%). */
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TR s = 0.0L;
            for (ptrdiff_t i = 0; i < j; ++i) s += ap[cs + (size_t)i] * x[i];
            s += nounit ? ap[cs + (size_t)j] * x[j] : x[j];
            y[j] += s;
        }
    } else {
        /* Diagonal (ap[cs]) is at the column start, so diag-first stays sequential. */
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TR s = nounit ? ap[cs] * x[j] : x[j];
            for (ptrdiff_t i = j + 1; i < n; ++i) s += ap[cs + (size_t)(i - j)] * x[i];
            y[j] += s;
        }
    }
}

/* Threaded out-of-place path. Returns 1 if it handled the call, 0 to fall back
 * to the serial reference. Kept in its own noinline function so the in-place
 * serial loops below compile in a clean register context (an inline threaded
 * block crowded the x87 allocation and slowed the UPPER NoTrans serial sweep
 * ~14%). */
__attribute__((noinline))
static ptrdiff_t etpmv_omp(bool upper, bool is_t, bool nounit, ptrdiff_t n, ptrdiff_t incx,
                     const TR *restrict ap, TR *restrict x)
{
    ptrdiff_t nthreads = 1;
    if (n >= 50 && !omp_in_parallel()) {
        nthreads = blas_omp_max_threads();
        if (n < 500 && nthreads > 2) nthreads = 2;
    }
    if (nthreads <= 1) return 0;

    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * (ptrdiff_t)incx : 0;
    TR *buf_all = (TR *)calloc((size_t)nthreads * (size_t)n, sizeof(TR));
    ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
    TR *xbuf = NULL;
    const TR *xptr = x;
    if (incx != 1 && buf_all && range_m) {
        xbuf = (TR *)malloc((size_t)n * sizeof(TR));
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
        TR *y = is_t ? buf_all : &buf_all[(size_t)tid * (size_t)n];
        ptrdiff_t m_from, m_to;
        if (upper) { m_from = range_m[nthreads - tid - 1]; m_to = range_m[nthreads - tid]; }
        else       { m_from = range_m[tid];               m_to = range_m[tid + 1]; }
        if (m_from < m_to) {
            if (is_t) tpmv_kernel_T(upper, nounit, n, m_from, m_to, ap, xptr, y);
            else      tpmv_kernel_N(upper, nounit, n, m_from, m_to, ap, xptr, y);
        }
    }
    if (!is_t) {  /* reduce private slots into slot 0 over the touched range */
        if (upper) {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_to_t = range_m[nthreads - t];
                const TR *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = 0; i < m_to_t; ++i) buf_all[i] += slot[i];
            }
        } else {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_from_t = range_m[t];
                const TR *slot = &buf_all[(size_t)t * (size_t)n];
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

static void etpmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TR *restrict ap,
    TR *restrict x, ptrdiff_t incx)
{
    const TR zero = 0.0L;
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (etpmv_omp(UPLO == 'U', TRANS == 'T', nounit, n, incx, ap, x)) return;
#endif

    /* Serial dispatch mirrors par's original nested (incx==1 / strided) x
     * (NoTrans / Trans) x (Upper / Lower) shape — kept intact so a layout
     * perturbation doesn't shuffle the jittery small-N strided leaves. The ONLY
     * change vs the original is the contiguous Upper-NoTrans leaf: it now calls
     * the byte-faithful ob-reference serial (epblas_etpmv_notrans_dsb), which is
     * DSB-resident on Coffee Lake (par's own in-place UNN sweep decoded via MITE
     * and lost ~2% at small N; par now ~parity/under ob there). Trans keeps par's
     * 2-chain forward dot (the ~15-20% win over ob's single-acc backward sweep). */
    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                epblas_etpmv_notrans_dsb(uplo, 'N', diag, n, ap, x, incx);
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const TR tmp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[i] += tmp * ap[k]; --k; }
                        if (nounit) x[j] *= ap[kk - (n - 1 - j)];
                    }
                    kk -= (n - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR tmp = x[j];
                    if (nounit) tmp *= ap[kk];
                    /* forward 2-chain over x[0..j-1]; ap[kk-j] pairs with x[0]
                     * (was a single-acc BACKWARD sweep). */
                    tmp += etpmv_dot2(&ap[kk - j], &x[0], j, 1);
                    x[j] = tmp;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR tmp = x[j];
                    if (nounit) tmp *= ap[kk];
                    /* forward 2-chain over x[j+1..n-1]; ap[kk+1] pairs with x[j+1] */
                    tmp += etpmv_dot2(&ap[kk + 1], &x[j + 1], n - 1 - j, 1);
                    x[j] = tmp;
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
                        const TR tmp = x[jx];
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
                        const TR tmp = x[jx];
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
                    TR tmp = x[jx];
                    if (nounit) tmp *= ap[kk];
                    /* forward 2-chain over rows 0..j-1 (x base kx, stride incx);
                     * ap[kk-j] pairs with row 0 (was backward, the flagged UTU). */
                    tmp += etpmv_dot2(&ap[kk - j], &x[kx], j, incx);
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR tmp = x[jx];
                    if (nounit) tmp *= ap[kk];
                    /* forward 2-chain over rows j+1..n-1 (x base jx+incx) */
                    tmp += etpmv_dot2(&ap[kk + 1], &x[jx + incx], n - 1 - j, incx);
                    x[jx] = tmp;
                    jx += incx;
                    kk += n - j;
                }
            }
        }
    }
}

EPBLAS_FACADE_TPMV(etpmv, TR)
