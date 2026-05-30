/*
 * etrsm_serial — kind10 (REAL(KIND=10) / long double) triangular solve,
 * pure single-thread worker half of the etrsm overlay (see
 * etrsm_kernel.h for the split rationale).
 *
 * Solves one of:
 *   op(A) · X = alpha · B          (SIDE='L')
 *   X · op(A) = alpha · B          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ, Aᴴ}; for real types Aᴴ ≡ Aᵀ. A is M×M (or N×N)
 * triangular (upper or lower; optionally unit-diagonal). B is overwritten
 * with the solution X.
 *
 * This file holds ALL the math: the column/row-range cores, the blocked
 * SIDE='L' worker, and the serial Fortran-ABI entry `etrsm_serial`. It
 * contains no `#pragma omp` — the threading lives entirely in
 * etrsm_parallel.c, which calls these same cores per team chunk.
 *
 * The rank-1-update loop structure matches upstream Netlib DTRSM so the
 * numerical behavior tracks the migrated archive to a tight tolerance.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

#include "etrsm_kernel.h"

typedef etrsm_T T;

/* Block size for the SIDE='L' blocked path. Env-tunable; default picked
 * empirically to match egemm's natural panel sizing. */
static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}
static int g_nb_trsm = 0;
int etrsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = env_int("ETRSM_NB", 64);
    return g_nb_trsm;
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Convenience accessors over column-major A and B. */
#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* The overlay's single-thread egemm worker (egemm_serial.c). We call it
 * — not the parallel egemm_ — because every trailing update here runs
 * inside the caller's `omp parallel` (blocked_dispatch in
 * etrsm_parallel.c); opening egemm's own nested team is what tripped the
 * libgomp barrier wedge (memory project-etrsm-omp4-wedge). egemm_ would
 * itself delegate here via its omp_in_parallel() guard; calling
 * egemm_serial directly makes the intent explicit and skips the check. */
extern void egemm_serial(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

/* ── SIDE = 'L': solve op(A) X = α B, A is M×M, B is M×N ───────── */

/* Column-range cores: serial work over columns j in [j_start, j_end).
 * Used both standalone (the parallel driver wraps them in its own
 * region) and inside the blocked path (where the outer parallel region
 * has already partitioned the column range). */

void etrsm_lln_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0L) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0L) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void etrsm_lun_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0L) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0L) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void etrsm_llt_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (int k = i + 1; k < M; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

void etrsm_lut_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (int k = 0; k < i; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

/* ── Blocked SIDE='L' worker ──────────────────────────────────────
 *
 * One column slice [j_start, j_end) of B, solved serially with `egemm`
 * trailing updates. The parallel driver partitions B's columns across a
 * team and calls this once per thread; the serial entry calls it once
 * over the whole column range. egemm calls run single-thread
 * (egemm_serial), so a 4-core TRSM is 4 cores each doing serial gemm on
 * their chunk — near-perfect scaling with one OMP setup for the whole
 * solve.
 */

/* Apply alpha to a column slice [j_start, j_end) of B in place. */
static inline void prescale_chunk(int j_start, int j_end, int M, T alpha,
                                  T *b, int ldb)
{
    if (alpha == 1.0L) return;
    if (alpha == 0.0L) {
        for (int j = j_start; j < j_end; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0L;
        return;
    }
    for (int j = j_start; j < j_end; ++j)
        for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
}

void etrsm_blocked_chunk(enum etrsm_variant V, int j_start, int j_end,
                         int M, int nb, T alpha,
                         const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, M, alpha, b, ldb);

    const T m_one = -1.0L, one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    /* The egemm call sees only this thread's column slice — operates
     * on B starting at column j_start, my_N columns wide. */
    T *B_chunk = &B_(0, j_start);

    if (V == LLN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                /* B[ic..ic+ib, j_start..j_end] -= A[ic..ic+ib, 0..ic] · B[0..ic, j_start..j_end] */
                egemm_serial(NN, NN, &ib, &my_N, &ic, &m_one,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            etrsm_lln_core(j_start, j_end, ib, 1.0L,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    } else if (V == LUN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                egemm_serial(NN, NN, &ib, &my_N, &trailing, &m_one,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            etrsm_lun_core(j_start, j_end, ib, 1.0L,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else if (V == LLT) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                egemm_serial(TN, NN, &ib, &my_N, &trailing, &m_one,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            etrsm_llt_core(j_start, j_end, ib, 1.0L,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else { /* LUT */
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                egemm_serial(TN, NN, &ib, &my_N, &ic, &m_one,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            etrsm_lut_core(j_start, j_end, ib, 1.0L,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    }
}

/* ── SIDE = 'R': solve X op(A) = α B, A is N×N, B is M×N ─────────
 *
 * R-side cores partition the M (row) axis: the j (column) loop walks the
 * diagonal serially (each B[:,j] depends on prior B[:,k]), but within
 * each step every row of B is processed identically. Each thread owns a
 * disjoint row slice [i_start, i_end) of B and reads shared A read-only —
 * race-free, no barriers needed. */

/* (R, L, N): solve X · A = α B, A lower triangular.
 * Equivalent to back-substitution on columns of B from j = N-1 down. */
void etrsm_rln_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0L) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0L / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

/* (R, U, N): solve X · A = α B, A upper triangular.
 * Forward-sub on columns of B from j = 0 up. */
void etrsm_run_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0L) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0L / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

/* (R, L, T): solve X · Aᵀ = α B, A lower. */
void etrsm_rlt_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = 1.0L / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0L) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* (R, U, T): solve X · Aᵀ = α B, A upper. */
void etrsm_rut_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = 1.0L / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0L) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* ── Pure-serial entry ────────────────────────────────────────────
 *
 * The full SIDE/UPLO/TRANSA dispatch, calling each core over its whole
 * range (no threading). etrsm_parallel.c's etrsm_ delegates here when
 * called from inside another routine's parallel region. */
void etrsm_serial(
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
    const char SIDE   = up(side);
    const char UPLO   = up(uplo);
    char TR           = up(transa);
    if (TR == 'C') TR = 'T';   /* real type: conj-trans ≡ trans */
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    /* alpha == 0 quick return: B becomes all zeros. */
    if (alpha == 0.0L) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0L;
        return;
    }

    if (SIDE == 'L') {
        /* Blocked path when M is large enough to amortize the egemm call
         * overhead — twice the block size; below that, blocking only adds
         * noise. */
        const int nb = etrsm_nb();
        const int use_blocked = (M >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) etrsm_blocked_chunk(LLN, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             etrsm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) etrsm_blocked_chunk(LUN, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             etrsm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) etrsm_blocked_chunk(LLT, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             etrsm_llt_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) etrsm_blocked_chunk(LUT, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
                else             etrsm_lut_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') etrsm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             etrsm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') etrsm_rlt_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             etrsm_rut_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
