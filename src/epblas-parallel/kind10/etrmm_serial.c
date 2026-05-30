/*
 * etrmm_serial — kind10 real (long double) triangular matrix-multiply,
 * single-thread. This TU owns ALL of the etrmm math: the eight scalar
 * column/row-range cores, the blocked per-chunk workers, and the
 * pure-serial Fortran-ABI entry. etrmm_parallel.c only orchestrates the
 * column/row partition across a team.
 *
 *   B := alpha · op(A) · B   (SIDE='L')   or   B := alpha · B · op(A) (R)
 *
 * Blocked path: diagonal scalar core in place + egemm_serial trailing
 * update. The trailing update runs through egemm_serial (NOT egemm_): when
 * a chunk worker runs inside the team etrmm_parallel.c opened, a nested
 * egemm team would trip the libgomp barrier wedge (project-etrsm-omp4-wedge).
 */

#include "etrmm_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef etrmm_T T;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}
static int g_nb_trmm = 0;
int etrmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = env_int("ETRMM_NB", 64);
    return g_nb_trmm;
}

extern void egemm_serial(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range scalar cores ─────────────────────── */

void etrmm_lln_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0L) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void etrmm_lun_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0L) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void etrmm_llt_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (int k = i + 1; k < M; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

void etrmm_lut_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (int k = 0; k < i; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range scalar cores ────────────────────────── */

void etrmm_rln_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0L)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0L) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void etrmm_run_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0L)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0L) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void etrmm_rlt_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0L) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0L)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void etrmm_rut_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0L) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0L)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* ── Blocked SIDE='L' chunk worker ────────────────────────────── */

void etrmm_blocked_chunk_L(enum etrmm_variant_L V, int j_start, int j_end,
                           int M, int nb, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;

    const T one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    T *B_chunk = &B_(0, j_start);

    if (V == ETRMM_LLN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            etrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                egemm_serial(NN, NN, &ib, &my_N, &ic, &alpha,
                             &A_(ic, 0), &lda,
                             B_chunk, &ldb, &one,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    } else if (V == ETRMM_LUN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            etrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                egemm_serial(NN, NN, &ib, &my_N, &trailing, &alpha,
                             &A_(ic, j0), &lda,
                             &B_chunk[j0], &ldb, &one,
                             &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else if (V == ETRMM_LLT) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            etrmm_llt_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                egemm_serial(TN, NN, &ib, &my_N, &trailing, &alpha,
                             &A_(i0, ic), &lda,
                             &B_chunk[i0], &ldb, &one,
                             &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else { /* ETRMM_LUT */
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            etrmm_lut_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                egemm_serial(TN, NN, &ib, &my_N, &ic, &alpha,
                             &A_(0, ic), &lda,
                             B_chunk, &ldb, &one,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    }
}

/* ── Blocked SIDE='R' chunk worker ────────────────────────────── */

void etrmm_blocked_chunk_R(enum etrmm_variant_R V, int i_start, int i_end,
                           int N, int nb, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    const T one = 1.0L;
    const char N_[1] = {'N'};
    const char T_[1] = {'T'};
    const int my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == ETRMM_RLN) {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            etrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                egemm_serial(N_, N_, &my_M, &jb, &trailing, &alpha,
                             &B_chunk[(size_t)k0 * ldb], &ldb,
                             &A_(k0, jc), &lda, &one,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    } else if (V == ETRMM_RUN) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            etrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                egemm_serial(N_, N_, &my_M, &jb, &jc, &alpha,
                             B_chunk, &ldb,
                             &A_(0, jc), &lda, &one,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else if (V == ETRMM_RLT) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            etrmm_rlt_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                egemm_serial(N_, T_, &my_M, &jb, &jc, &alpha,
                             B_chunk, &ldb,
                             &A_(jc, 0), &lda, &one,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else { /* ETRMM_RUT */
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            etrmm_rut_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                egemm_serial(N_, T_, &my_M, &jb, &trailing, &alpha,
                             &B_chunk[(size_t)k0 * ldb], &ldb,
                             &A_(jc, k0), &lda, &one,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    }
}

/* ── Serial entry ─────────────────────────────────────────────── */

void etrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);
    char TR = (char)toupper((unsigned char)*transa);
    if (TR == 'C') TR = 'T';
    const int nounit = ((char)toupper((unsigned char)*diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0L) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0L;
        return;
    }

    const int nb = etrmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
        enum etrmm_variant_L V = (TR == 'N')
            ? (UPLO == 'L' ? ETRMM_LLN : ETRMM_LUN)
            : (UPLO == 'L' ? ETRMM_LLT : ETRMM_LUT);
        if (use_blocked) {
            etrmm_blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
        } else {
            switch (V) {
            case ETRMM_LLN: etrmm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit); break;
            case ETRMM_LUN: etrmm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit); break;
            case ETRMM_LLT: etrmm_llt_core(0, N, M, alpha, a, lda, b, ldb, nounit); break;
            case ETRMM_LUT: etrmm_lut_core(0, N, M, alpha, a, lda, b, ldb, nounit); break;
            }
        }
    } else {
        const int use_blocked = (N >= 2 * nb);
        enum etrmm_variant_R V = (TR == 'N')
            ? (UPLO == 'L' ? ETRMM_RLN : ETRMM_RUN)
            : (UPLO == 'L' ? ETRMM_RLT : ETRMM_RUT);
        if (use_blocked) {
            etrmm_blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
        } else {
            switch (V) {
            case ETRMM_RLN: etrmm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit); break;
            case ETRMM_RUN: etrmm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit); break;
            case ETRMM_RLT: etrmm_rlt_core(0, M, N, alpha, a, lda, b, ldb, nounit); break;
            case ETRMM_RUT: etrmm_rut_core(0, M, N, alpha, a, lda, b, ldb, nounit); break;
            }
        }
    }
}

#undef A_
#undef B_
