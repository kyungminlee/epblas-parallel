/*
 * ytrsm_serial — kind10 complex (COMPLEX(KIND=10) / _Complex long double)
 * triangular solve, single-thread. This TU owns ALL of the ytrsm math
 * (column/row-range cores + the blocked SIDE='L' worker); ytrsm_parallel.c
 * only orchestrates threads over these same pieces.
 *
 * Same scaffolding as the real etrsm with TRANSA='C' (conjugate transpose)
 * handled as a distinct case from 'T'. No SIMD path — x87 long-double has
 * no SIMD register file on x86_64. The SIDE='L' blocked path runs its
 * trailing-matrix update through ygemm_serial (NOT ygemm_): when this code
 * runs inside the team ytrsm_parallel.c opened, calling the parallel ygemm_
 * would open a nested team and trip the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 *
 * Fortran ABI (ytrsm_serial mirrors ytrsm_ exactly):
 *   - scalars by pointer; complex scalar = pointer to (re, im) pair
 *   - character args followed by hidden trailing `size_t` lengths
 *   - COMPLEX(KIND=10) ↔ `_Complex long double` (32 bytes on x86-64)
 */

#include "ytrsm_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ytrsm_T T;

static ptrdiff_t env_int(const char *name, ptrdiff_t dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    ptrdiff_t v = atoi(s);
    return v > 0 ? v : dflt;
}

static ptrdiff_t g_nb_trsm = 0;
ptrdiff_t ytrsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = env_int("YTRSM_NB", 64);
    return g_nb_trsm;
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;
static const T NEG_ONE = -1.0L + 0.0Li;

static inline T cconj(T a) { return ~a; }

/* SIDE='L' blocked trailing update — runs inside the calling thread, so it
 * must use the single-thread ygemm (see file header). */
extern void ygemm_serial(
    const char *transa, const char *transb,
    const ptrdiff_t *m, const ptrdiff_t *n, const ptrdiff_t *k,
    const T *alpha,
    const T *a, const ptrdiff_t *lda,
    const T *b, const ptrdiff_t *ldb,
    const T *beta,
    T *c, const ptrdiff_t *ldc,
    size_t transa_len, size_t transb_len);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, ptrdiff_t lda, ptrdiff_t row, ptrdiff_t col, ptrdiff_t conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── Scalar column-range cores ───────────────────────────────── */

void ytrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void ytrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

/* (L, L, T or C): inner-product form on op(A)ᵀ where op may conj. */
void ytrsm_llTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = i + 1; k < M; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* (L, U, T or C). */
void ytrsm_luTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = 0; k < i; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ───────────────────────────────────
 *
 * The j (column) loop walks the diagonal serially (each B[:,j] depends
 * on prior B[:,k]), but within each step every row of B is processed
 * identically. Each thread owns a disjoint row slice [i_start, i_end)
 * of B and reads shared A read-only — race-free, no barriers needed. */

void ytrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = N - 1; j >= 0; --j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void ytrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    for (ptrdiff_t j = 0; j < N; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void ytrsm_rlTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void ytrsm_ruTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag)
{
    for (ptrdiff_t k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* ── Blocked SIDE='L' worker ──────────────────────────────────── */

static void prescale_chunk(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha, T *b, ptrdiff_t ldb)
{
    if (alpha == ONE) return;
    if (alpha == ZERO) {
        for (ptrdiff_t j = j_start; j < j_end; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }
    for (ptrdiff_t j = j_start; j < j_end; ++j)
        for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
}

void ytrsm_blocked_chunk(enum ytrsm_variant V, ptrdiff_t j_start, ptrdiff_t j_end,
                         ptrdiff_t M, ptrdiff_t nb, T alpha,
                         const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, M, alpha, b, ldb);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    T *B_chunk = &B_(0, j_start);

    if (V == YLLN) {
        for (ptrdiff_t ic = 0; ic < M; ic += nb) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                ygemm_serial(NN, NN, &ib, &my_N, &ic, &NEG_ONE,
                             &A_(ic, 0), &lda,
                             B_chunk, &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_lln_core(j_start, j_end, ib, ONE,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    } else if (V == YLUN) {
        ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            const ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const ptrdiff_t j0 = ic + ib;
                ygemm_serial(NN, NN, &ib, &my_N, &trailing, &NEG_ONE,
                             &A_(ic, j0), &lda,
                             &B_chunk[j0], &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_lun_core(j_start, j_end, ib, ONE,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else if (V == YLLT || V == YLLC) {
        const ptrdiff_t conj_flag = (V == YLLC) ? 1 : 0;
        const char *trans_gemm = conj_flag ? CN : TN;
        ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            const ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const ptrdiff_t i0 = ic + ib;
                ygemm_serial(trans_gemm, NN, &ib, &my_N, &trailing, &NEG_ONE,
                             &A_(i0, ic), &lda,
                             &B_chunk[i0], &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_llTC_core(j_start, j_end, ib, ONE,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb,
                            nounit, conj_flag);
            ic -= nb;
        }
    } else { /* YLUT or YLUC */
        const ptrdiff_t conj_flag = (V == YLUC) ? 1 : 0;
        const char *trans_gemm = conj_flag ? CN : TN;
        for (ptrdiff_t ic = 0; ic < M; ic += nb) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                ygemm_serial(trans_gemm, NN, &ib, &my_N, &ic, &NEG_ONE,
                             &A_(0, ic), &lda,
                             B_chunk, &ldb, &ONE,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_luTC_core(j_start, j_end, ib, ONE,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb,
                            nounit, conj_flag);
        }
    }
}

/* ── Blocked SIDE='R' worker ───────────────────────────────────────
 *
 * The complex twin of the SIDE='L' blocked path, transposed in role: A is
 * N×N, B is M×N, and the triangular axis being blocked is the column (N)
 * axis. The diagonal jb×jb block is solved with the naive R core; every
 * earlier/later solved column block is folded in by a single ygemm_serial
 * (the bulk of the FLOPs, at packed-GEMM speed). Threads own disjoint row
 * bands [i_start, i_end) of B and read shared A read-only, so each band
 * runs this independently with no barrier (same contract as the cores).
 *
 * Direction (matches OpenBLAS trsm_R): forward (column blocks ascending,
 * each solved from the already-solved blocks to its left) for the
 * upper-NoTrans and lower-Trans shapes; backward otherwise. */
void ytrsm_R_blocked_chunk(ptrdiff_t upper, ptrdiff_t trans, ptrdiff_t conj,
                           ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, ptrdiff_t nb, T alpha,
                           const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    if (i_end <= i_start) return;
    /* Prescale this row band by alpha once; the cores then run alpha=ONE. */
    if (alpha != ONE)
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;

    const ptrdiff_t m_band = i_end - i_start;
    const char NN[1] = {'N'};
    const char TC[1] = {conj ? 'C' : 'T'};
    const ptrdiff_t forward = (upper && !trans) || (!upper && trans);

    if (forward) {
        for (ptrdiff_t jc = 0; jc < N; jc += nb) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            if (jc > 0) {
                /* B[:,jc:jc+jb] -= B[:,0:jc] · op(A)[0:jc, jc:jc+jb]. */
                ygemm_serial(NN, trans ? TC : NN, &m_band, &jb, &jc, &NEG_ONE,
                             &B_(i_start, 0), &ldb,
                             trans ? &A_(jc, 0) : &A_(0, jc), &lda, &ONE,
                             &B_(i_start, jc), &ldb, 1, 1);
            }
            if (trans)  /* lower · op = lower-Trans/Conj */
                ytrsm_rlTC_core(i_start, i_end, jb, ONE,
                                &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj);
            else        /* upper · NoTrans */
                ytrsm_run_core(i_start, i_end, jb, ONE,
                               &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
        }
    } else {
        ptrdiff_t jc = ((N - 1) / nb) * nb;
        for (; jc >= 0; jc -= nb) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            const ptrdiff_t trail = N - (jc + jb);
            if (trail > 0) {
                const ptrdiff_t j0 = jc + jb;
                /* B[:,jc:jc+jb] -= B[:,j0:N] · op(A)[j0:N, jc:jc+jb]. */
                ygemm_serial(NN, trans ? TC : NN, &m_band, &jb, &trail, &NEG_ONE,
                             &B_(i_start, j0), &ldb,
                             trans ? &A_(jc, j0) : &A_(j0, jc), &lda, &ONE,
                             &B_(i_start, jc), &ldb, 1, 1);
            }
            if (trans)  /* upper · op = upper-Trans/Conj */
                ytrsm_ruTC_core(i_start, i_end, jb, ONE,
                                &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj);
            else        /* lower · NoTrans */
                ytrsm_rln_core(i_start, i_end, jb, ONE,
                               &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
        }
    }
}

/* ── Single-thread entry ──────────────────────────────────────── */

void ytrsm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const ptrdiff_t *m_, const ptrdiff_t *n_,
    const T *alpha_,
    const T *a, const ptrdiff_t *lda_,
    T *b, const ptrdiff_t *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const ptrdiff_t M = *m_, N = *n_;
    const ptrdiff_t lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    const char TR = up(transa);
    const ptrdiff_t nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    if (SIDE == 'L') {
        const ptrdiff_t use_blocked = (M >= 2 * ytrsm_nb());
        const ptrdiff_t nb = ytrsm_nb();
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) ytrsm_blocked_chunk(YLLN, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) ytrsm_blocked_chunk(YLUN, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            }
        } else if (TR == 'T') {
            if (UPLO == 'L') {
                if (use_blocked) ytrsm_blocked_chunk(YLLT, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0);
            } else {
                if (use_blocked) ytrsm_blocked_chunk(YLUT, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0);
            }
        } else { /* 'C' */
            if (UPLO == 'L') {
                if (use_blocked) ytrsm_blocked_chunk(YLLC, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 1);
            } else {
                if (use_blocked) ytrsm_blocked_chunk(YLUC, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 1);
            }
        }
    } else {
        const ptrdiff_t use_blocked = (N >= 2 * ytrsm_nb());
        const ptrdiff_t nb = ytrsm_nb();
        const ptrdiff_t upper = (UPLO == 'U');
        const ptrdiff_t trans = (TR != 'N');
        const ptrdiff_t conj  = (TR == 'C');
        if (use_blocked) {
            ytrsm_R_blocked_chunk(upper, trans, conj, 0, M, N, nb, alpha,
                                  a, lda, b, ldb, nounit);
        } else if (TR == 'N') {
            if (UPLO == 'L') ytrsm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') ytrsm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0);
            else             ytrsm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') ytrsm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 1);
            else             ytrsm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 1);
        }
    }
}

#undef A_
#undef B_
