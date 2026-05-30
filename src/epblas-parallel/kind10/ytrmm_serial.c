/*
 * ytrmm_serial — kind10 complex (COMPLEX(KIND=10) / _Complex long double)
 * triangular multiply, single-thread. This TU owns ALL of the ytrmm math
 * (column/row-range cores + the blocked SIDE='L'/'R' workers);
 * ytrmm_parallel.c only orchestrates threads over these same pieces.
 *
 * Same blocked scaffold as the real etrmm with conjugate-transpose 'C'
 * handled as a distinct case from 'T'. The blocked path runs its
 * trailing-matrix update through ygemm_serial (NOT ygemm_): when this code
 * runs inside the team ytrmm_parallel.c opened, calling the parallel ygemm_
 * would open a nested team and trip the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 *
 * Fortran ABI (ytrmm_serial mirrors ytrmm_ exactly):
 *   - scalars by pointer; complex scalar = pointer to (re, im) pair
 *   - character args followed by hidden trailing `size_t` lengths
 *   - COMPLEX(KIND=10) ↔ `_Complex long double` (32 bytes on x86-64)
 */

#include "ytrmm_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef ytrmm_T T;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}
static int g_nb_trmm = 0;
int ytrmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = env_int("YTRMM_NB", 32);
    return g_nb_trmm;
}

extern void ygemm_serial(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static inline T cconj(T a) { return ~a; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── Scalar column-range cores (SIDE='L') ───────────────────────── */

void ytrmm_lln_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void ytrmm_lun_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void ytrmm_llTC_core(int j_start, int j_end, int M, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (int k = i + 1; k < M; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

void ytrmm_luTC_core(int j_start, int j_end, int M, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (int k = 0; k < i; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── Scalar row-range cores (SIDE='R') ──────────────────────────── */

void ytrmm_rln_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void ytrmm_run_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void ytrmm_rlTC_core(int i_start, int i_end, int N, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void ytrmm_ruTC_core(int i_start, int i_end, int N, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* ── Blocked SIDE='L' worker ────────────────────────────────────── */

void ytrmm_blocked_chunk_L(enum ytrmm_variant_L V, int j_start, int j_end,
                           int M, int nb, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    T *B_chunk = &B_(0, j_start);

    if (V == YLLN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                ygemm_serial(NN, NN, &ib, &my_N, &ic, &alpha,
                             &A_(ic, 0), &lda,
                             B_chunk, &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    } else if (V == YLUN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                ygemm_serial(NN, NN, &ib, &my_N, &trailing, &alpha,
                             &A_(ic, j0), &lda,
                             &B_chunk[j0], &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else if (V == YLLT || V == YLLC) {
        const int conj_flag = (V == YLLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_llTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                ygemm_serial(gemm_trans, NN, &ib, &my_N, &trailing, &alpha,
                             &A_(i0, ic), &lda,
                             &B_chunk[i0], &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else { /* YLUT or YLUC */
        const int conj_flag = (V == YLUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_luTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            if (ic > 0) {
                ygemm_serial(gemm_trans, NN, &ib, &my_N, &ic, &alpha,
                             &A_(0, ic), &lda,
                             B_chunk, &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    }
}

/* ── Blocked SIDE='R' worker ────────────────────────────────────── */

void ytrmm_blocked_chunk_R(enum ytrmm_variant_R V, int i_start, int i_end,
                           int N, int nb, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    const int my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == YRLN) {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                ygemm_serial(NN, NN, &my_M, &jb, &trailing, &alpha,
                             &B_chunk[(size_t)k0 * ldb], &ldb,
                             &A_(k0, jc), &lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    } else if (V == YRUN) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                ygemm_serial(NN, NN, &my_M, &jb, &jc, &alpha,
                             B_chunk, &ldb,
                             &A_(0, jc), &lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else if (V == YRLT || V == YRLC) {
        const int conj_flag = (V == YRLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_rlTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            if (jc > 0) {
                ygemm_serial(NN, gemm_trans, &my_M, &jb, &jc, &alpha,
                             B_chunk, &ldb,
                             &A_(jc, 0), &lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else { /* YRUT or YRUC */
        const int conj_flag = (V == YRUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_ruTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                ygemm_serial(NN, gemm_trans, &my_M, &jb, &trailing, &alpha,
                             &B_chunk[(size_t)k0 * ldb], &ldb,
                             &A_(jc, k0), &lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    }
}

/* ── Single-thread entry ──────────────────────────────────────── */

void ytrmm_serial(
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
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    const char TR = up(transa);
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    const int nb = ytrmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
        enum ytrmm_variant_L V;
        if (TR == 'N')      V = (UPLO == 'L') ? YLLN : YLUN;
        else if (TR == 'T') V = (UPLO == 'L') ? YLLT : YLUT;
        else                V = (UPLO == 'L') ? YLLC : YLUC;
        if (use_blocked) {
            ytrmm_blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
        } else {
            switch (V) {
            case YLLN: ytrmm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit); break;
            case YLUN: ytrmm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit); break;
            case YLLT: ytrmm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0); break;
            case YLUT: ytrmm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0); break;
            case YLLC: ytrmm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 1); break;
            case YLUC: ytrmm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 1); break;
            }
        }
    } else {
        const int use_blocked = (N >= 2 * nb);
        enum ytrmm_variant_R V;
        if (TR == 'N')      V = (UPLO == 'L') ? YRLN : YRUN;
        else if (TR == 'T') V = (UPLO == 'L') ? YRLT : YRUT;
        else                V = (UPLO == 'L') ? YRLC : YRUC;
        if (use_blocked) {
            ytrmm_blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
        } else {
            switch (V) {
            case YRLN: ytrmm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit); break;
            case YRUN: ytrmm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit); break;
            case YRLT: ytrmm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0); break;
            case YRUT: ytrmm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0); break;
            case YRLC: ytrmm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 1); break;
            case YRUC: ytrmm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 1); break;
            }
        }
    }
}

#undef A_
#undef B_
