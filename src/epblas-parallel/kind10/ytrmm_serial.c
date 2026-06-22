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
#include <stddef.h>

typedef ytrmm_T T;

static ptrdiff_t g_nb_trmm = 0;
ptrdiff_t ytrmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = 32;
    return g_nb_trmm;
}

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta,
    T *c, ptrdiff_t ldc);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static inline T cconj(T a) { return ~a; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, ptrdiff_t lda, ptrdiff_t row, ptrdiff_t col, ptrdiff_t conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── Scalar column-range cores (SIDE='L') ───────────────────────── */

void ytrmm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (ptrdiff_t i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void ytrmm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void ytrmm_llTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (ptrdiff_t k = i + 1; k < M; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

void ytrmm_luTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (ptrdiff_t k = 0; k < i; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── Scalar row-range cores (SIDE='R') ──────────────────────────── */

void ytrmm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (ptrdiff_t k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void ytrmm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void ytrmm_rlTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t k = N - 1; k >= 0; --k) {
        for (ptrdiff_t j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void ytrmm_ruTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t k = 0; k < N; ++k) {
        for (ptrdiff_t j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* ── Blocked SIDE='L' worker ────────────────────────────────────── */

void ytrmm_blocked_chunk_L(enum ytrmm_variant_L V, ptrdiff_t j_start, ptrdiff_t j_end,
                           ptrdiff_t M, ptrdiff_t nb, T alpha,
                           const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;

    T *B_chunk = &B_(0, j_start);

    if (V == YLLN) {
        ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                ygemm_serial('N', 'N', ib, my_N, ic, &alpha,
                             &A_(ic, 0), lda,
                             B_chunk, ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
            ic -= nb;
        }
    } else if (V == YLUN) {
        for (ptrdiff_t ic = 0; ic < M; ic += nb) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const ptrdiff_t j0 = ic + ib;
                ygemm_serial('N', 'N', ib, my_N, trailing, &alpha,
                             &A_(ic, j0), lda,
                             &B_chunk[j0], ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
        }
    } else if (V == YLLT || V == YLLC) {
        const ptrdiff_t conj_flag = (V == YLLC) ? 1 : 0;
        const char gemm_trans = conj_flag ? 'C' : 'T';
        for (ptrdiff_t ic = 0; ic < M; ic += nb) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_llTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            const ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const ptrdiff_t i0 = ic + ib;
                ygemm_serial(gemm_trans, 'N', ib, my_N, trailing, &alpha,
                             &A_(i0, ic), lda,
                             &B_chunk[i0], ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
        }
    } else { /* YLUT or YLUC */
        const ptrdiff_t conj_flag = (V == YLUC) ? 1 : 0;
        const char gemm_trans = conj_flag ? 'C' : 'T';
        ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_luTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            if (ic > 0) {
                ygemm_serial(gemm_trans, 'N', ib, my_N, ic, &alpha,
                             &A_(0, ic), lda,
                             B_chunk, ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
            ic -= nb;
        }
    }
}

/* ── Blocked SIDE='R' worker ────────────────────────────────────── */

void ytrmm_blocked_chunk_R(enum ytrmm_variant_R V, ptrdiff_t i_start, ptrdiff_t i_end,
                           ptrdiff_t N, ptrdiff_t nb, T alpha,
                           const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == YRLN) {
        for (ptrdiff_t jc = 0; jc < N; jc += nb) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const ptrdiff_t trailing = N - (jc + jb);
            if (trailing > 0) {
                const ptrdiff_t k0 = jc + jb;
                ygemm_serial('N', 'N', my_M, jb, trailing, &alpha,
                             &B_chunk[(size_t)k0 * ldb], ldb,
                             &A_(k0, jc), lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], ldb);
            }
        }
    } else if (V == YRUN) {
        ptrdiff_t jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                ygemm_serial('N', 'N', my_M, jb, jc, &alpha,
                             B_chunk, ldb,
                             &A_(0, jc), lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], ldb);
            }
            jc -= nb;
        }
    } else if (V == YRLT || V == YRLC) {
        const ptrdiff_t conj_flag = (V == YRLC) ? 1 : 0;
        const char gemm_trans = conj_flag ? 'C' : 'T';
        ptrdiff_t jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_rlTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            if (jc > 0) {
                ygemm_serial('N', gemm_trans, my_M, jb, jc, &alpha,
                             B_chunk, ldb,
                             &A_(jc, 0), lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], ldb);
            }
            jc -= nb;
        }
    } else { /* YRUT or YRUC */
        const ptrdiff_t conj_flag = (V == YRUC) ? 1 : 0;
        const char gemm_trans = conj_flag ? 'C' : 'T';
        for (ptrdiff_t jc = 0; jc < N; jc += nb) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_ruTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            const ptrdiff_t trailing = N - (jc + jb);
            if (trailing > 0) {
                const ptrdiff_t k0 = jc + jb;
                ygemm_serial('N', gemm_trans, my_M, jb, trailing, &alpha,
                             &B_chunk[(size_t)k0 * ldb], ldb,
                             &A_(jc, k0), lda, &ONE,
                             &B_chunk[(size_t)jc * ldb], ldb);
            }
        }
    }
}

/* ── Single-thread entry ──────────────────────────────────────── */

void ytrmm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    const char TR = up(&transa);
    const ptrdiff_t nounit = (up(&diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    const ptrdiff_t nb = ytrmm_nb();

    if (SIDE == 'L') {
        const ptrdiff_t use_blocked = (M >= 2 * nb);
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
        const ptrdiff_t use_blocked = (N >= 2 * nb);
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
