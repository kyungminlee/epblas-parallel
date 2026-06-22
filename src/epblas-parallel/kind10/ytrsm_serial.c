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
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ytrsm_T T;

static ptrdiff_t g_nb_trsm = 0;
ptrdiff_t ytrsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = 64;
    return g_nb_trsm;
}


static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;
static const T NEG_ONE = -1.0L + 0.0Li;

static inline T cconj(T a) { return ~a; }

/* SIDE='L' blocked trailing update — runs inside the calling thread, so it
 * must use the single-thread ygemm (see file header). */
extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta,
    T *c, ptrdiff_t ldc);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, ptrdiff_t lda, ptrdiff_t row, ptrdiff_t col, bool conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── Scalar column-range cores ───────────────────────────────── */

void ytrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < m; ++k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = k + 1; i < m; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void ytrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = m - 1; k >= 0; --k) {
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
void ytrsm_llTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = m - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = i + 1; k < m; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* (L, U, T or C). */
void ytrsm_luTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < m; ++i) {
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

void ytrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = j + 1; k < n; ++k) {
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

void ytrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = 0; j < n; ++j) {
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

void ytrsm_rlTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = 0; k < n; ++k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = k + 1; j < n; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void ytrsm_ruTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = n - 1; k >= 0; --k) {
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

static void prescale_chunk(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha, T *b, ptrdiff_t ldb)
{
    if (alpha == ONE) return;
    if (alpha == ZERO) {
        for (ptrdiff_t j = j_start; j < j_end; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = ZERO;
        return;
    }
    for (ptrdiff_t j = j_start; j < j_end; ++j)
        for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
}

void ytrsm_blocked_chunk(enum ytrsm_variant V, ptrdiff_t j_start, ptrdiff_t j_end,
                         ptrdiff_t m, ptrdiff_t nb, T alpha,
                         const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    const ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, m, alpha, b, ldb);

    T *B_chunk = &B_(0, j_start);

    if (V == YLLN) {
        for (ptrdiff_t ic = 0; ic < m; ic += nb) {
            const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            if (ic > 0) {
                ygemm_serial('N', 'N', ib, my_N, ic, &NEG_ONE,
                             &A_(ic, 0), lda,
                             B_chunk, ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
            ytrsm_lln_core(j_start, j_end, ib, ONE,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    } else if (V == YLUN) {
        ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            const ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const ptrdiff_t j0 = ic + ib;
                ygemm_serial('N', 'N', ib, my_N, trailing, &NEG_ONE,
                             &A_(ic, j0), lda,
                             &B_chunk[j0], ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
            ytrsm_lun_core(j_start, j_end, ib, ONE,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else if (V == YLLT || V == YLLC) {
        const bool conj_flag = (V == YLLC) ? 1 : 0;
        const char trans_gemm = conj_flag ? 'C' : 'T';
        ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            const ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const ptrdiff_t i0 = ic + ib;
                ygemm_serial(trans_gemm, 'N', ib, my_N, trailing, &NEG_ONE,
                             &A_(i0, ic), lda,
                             &B_chunk[i0], ldb, &ONE,
                             &B_chunk[ic], ldb);
            }
            ytrsm_llTC_core(j_start, j_end, ib, ONE,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb,
                            nounit, conj_flag);
            ic -= nb;
        }
    } else { /* YLUT or YLUC */
        const bool conj_flag = (V == YLUC) ? 1 : 0;
        const char trans_gemm = conj_flag ? 'C' : 'T';
        for (ptrdiff_t ic = 0; ic < m; ic += nb) {
            const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            if (ic > 0) {
                ygemm_serial(trans_gemm, 'N', ib, my_N, ic, &NEG_ONE,
                             &A_(0, ic), lda,
                             B_chunk, ldb, &ONE,
                             &B_chunk[ic], ldb);
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
void ytrsm_R_blocked_chunk(bool upper, bool trans, bool conj,
                           ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, ptrdiff_t nb, T alpha,
                           const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    if (i_end <= i_start) return;
    /* Prescale this row band by alpha once; the cores then run alpha=ONE. */
    if (alpha != ONE)
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;

    const ptrdiff_t m_band = i_end - i_start;
    const ptrdiff_t forward = (upper && !trans) || (!upper && trans);

    if (forward) {
        for (ptrdiff_t jc = 0; jc < n; jc += nb) {
            const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
            if (jc > 0) {
                /* B[:,jc:jc+jb] -= B[:,0:jc] · op(A)[0:jc, jc:jc+jb]. */
                ygemm_serial('N', trans ? (conj ? 'C' : 'T') : 'N', m_band, jb, jc, &NEG_ONE,
                             &B_(i_start, 0), ldb,
                             trans ? &A_(jc, 0) : &A_(0, jc), lda, &ONE,
                             &B_(i_start, jc), ldb);
            }
            if (trans)  /* lower · op = lower-Trans/Conj */
                ytrsm_rlTC_core(i_start, i_end, jb, ONE,
                                &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj);
            else        /* upper · NoTrans */
                ytrsm_run_core(i_start, i_end, jb, ONE,
                               &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
        }
    } else {
        ptrdiff_t jc = ((n - 1) / nb) * nb;
        for (; jc >= 0; jc -= nb) {
            const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
            const ptrdiff_t trail = n - (jc + jb);
            if (trail > 0) {
                const ptrdiff_t j0 = jc + jb;
                /* B[:,jc:jc+jb] -= B[:,j0:N] · op(A)[j0:N, jc:jc+jb]. */
                ygemm_serial('N', trans ? (conj ? 'C' : 'T') : 'N', m_band, jb, trail, &NEG_ONE,
                             &B_(i_start, j0), ldb,
                             trans ? &A_(jc, j0) : &A_(j0, jc), lda, &ONE,
                             &B_(i_start, jc), ldb);
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
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE = blas_up(side);
    const char UPLO = blas_up(uplo);
    const char TR = blas_up(transa);
    const bool nounit = (blas_up(diag) != 'U');

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = ZERO;
        return;
    }

    if (SIDE == 'L') {
        const ptrdiff_t use_blocked = (m >= 2 * ytrsm_nb());
        const ptrdiff_t nb = ytrsm_nb();
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) ytrsm_blocked_chunk(YLLN, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lln_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) ytrsm_blocked_chunk(YLUN, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lun_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            }
        } else if (TR == 'T') {
            if (UPLO == 'L') {
                if (use_blocked) ytrsm_blocked_chunk(YLLT, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
            } else {
                if (use_blocked) ytrsm_blocked_chunk(YLUT, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_luTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
            }
        } else { /* 'C' */
            if (UPLO == 'L') {
                if (use_blocked) ytrsm_blocked_chunk(YLLC, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, 1);
            } else {
                if (use_blocked) ytrsm_blocked_chunk(YLUC, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_luTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, 1);
            }
        }
    } else {
        const ptrdiff_t use_blocked = (n >= 2 * ytrsm_nb());
        const ptrdiff_t nb = ytrsm_nb();
        const bool upper = (UPLO == 'U');
        const bool trans = (TR != 'N');
        const bool conj  = (TR == 'C');
        if (use_blocked) {
            ytrsm_R_blocked_chunk(upper, trans, conj, 0, m, n, nb, alpha,
                                  a, lda, b, ldb, nounit);
        } else if (TR == 'N') {
            if (UPLO == 'L') ytrsm_rln_core(0, m, n, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_run_core(0, m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') ytrsm_rlTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
            else             ytrsm_ruTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') ytrsm_rlTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, 1);
            else             ytrsm_ruTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, 1);
        }
    }
}

#undef A_
#undef B_
