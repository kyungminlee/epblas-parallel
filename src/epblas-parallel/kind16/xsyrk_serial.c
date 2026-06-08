/*
 * xsyrk_serial — kind16 complex (__complex128) symmetric rank-k,
 * single-thread. This TU owns ALL of the xsyrk math; xsyrk_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * TRANS ∈ {N, T}. Complex syrk does NOT conjugate (see xherk). Blocked:
 * scalar diagonal + xgemm trailing. The trailing update runs through
 * xgemm_serial_ (NOT xgemm_): when xsyrk_block runs inside the team
 * xsyrk_parallel.c opened, a nested xgemm team would open a region inside a
 * region. Mirrors the kind10 ysyrk overlay.
 *
 * The block dims threaded into the trailing GEMMs are bridged to xgemm's int
 * Fortran ABI by xgemm_s() below.
 */

#include "xsyrk_kernel.h"
#include "xgemm_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef xsyrk_T T;

char xsyrk_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

char xsyrk_trans(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static ptrdiff_t g_xsyrk_nb = 0;
ptrdiff_t xsyrk_nb(void) {
    if (g_xsyrk_nb == 0) {
        g_xsyrk_nb = 32;
        const char *s = getenv("XSYRK_NB");
        if (s && *s) { ptrdiff_t v = atoi(s); if (v > 0) g_xsyrk_nb = v; }
    }
    return g_xsyrk_nb;
}

/* Bridge the ptrdiff_t block dims of the trailing update to xgemm_serial_'s
 * int Fortran ABI (block sizes are bounded by N/K, which arrive as int). */
static inline void xgemm_s(const char *ta, const char *tb,
                           ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, const T *alpha,
                           const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                           const T *beta, T *c, ptrdiff_t ldc)
{
    int mi = (int)m, ni = (int)n, ki = (int)k;
    int ldai = (int)lda, ldbi = (int)ldb, ldci = (int)ldc;
    xgemm_serial_(ta, tb, &mi, &ni, &ki, alpha, a, &ldai, b, &ldbi, beta, c, &ldci, 1, 1);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static void syrk_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t K, T alpha,
                          const T *restrict a, ptrdiff_t lda,
                          T *restrict c, ptrdiff_t ldc,
                          char UPLO, char TR)
{
    if (TR == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < K; ++l) {
                const T ajl = A_(j, l);
                if (ajl != ZERO) {
                    const T t = alpha * ajl;
                    for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] += t * A_(i, l);
                }
            }
        }
    } else {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            const T *Aj = a + (size_t)j * lda;
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                T s = ZERO;
                for (ptrdiff_t l = 0; l < K; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void xsyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, T beta,
                      T *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void xsyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K, T alpha, T beta,
                 const T *a, ptrdiff_t lda, T *c, ptrdiff_t ldc, char UPLO, char TR)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    syrk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TR == 'N') {
                xgemm_s(NN, TN, trailing, jb, K, &alpha,
                        &A_(j0, 0), lda, &A_(jc, 0), lda,
                        &ONE, &C_(j0, jc), ldc);
            } else {
                xgemm_s(TN, NN, trailing, jb, K, &alpha,
                        &A_(0, j0), lda, &A_(0, jc), lda,
                        &ONE, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                xgemm_s(NN, TN, jc, jb, K, &alpha,
                        &A_(0, 0), lda, &A_(jc, 0), lda,
                        &ONE, &C_(0, jc), ldc);
            } else {
                xgemm_s(TN, NN, jc, jb, K, &alpha,
                        &A_(0, 0), lda, &A_(0, jc), lda,
                        &ONE, &C_(0, jc), ldc);
            }
        }
    }
}

void xsyrk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR   = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
        xsyrk_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = xsyrk_nb();
    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        xsyrk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef C_
