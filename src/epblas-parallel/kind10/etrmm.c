/*
 * etrmm — kind10 (REAL(KIND=10) / `long double`) triangular multiply.
 *
 * Computes one of:
 *   B := alpha · op(A) · B          (SIDE='L')
 *   B := alpha · B · op(A)          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ, Aᴴ}; for real types Aᴴ ≡ Aᵀ. A is M×M (or
 * N×N) triangular (upper or lower; optionally unit-diagonal). B is
 * overwritten in place.
 *
 * Blocked Goto-style: for SIDE='L', for each M-block (ib×ib diagonal
 * + ib×rest off-diagonal):
 *
 *   1. Apply the diagonal block in place via the scalar trmm core
 *      (B_block := alpha · A_diag · B_block).
 *   2. Add the off-diagonal contribution as a single egemm call
 *      (B_block += alpha · A_offdiag · B_other) with beta=1.
 *
 * Iteration direction depends on UPLO/TRANSA so each block reads the
 * still-untouched portion of B before overwriting itself. For SIDE='R'
 * the same algorithm runs on column blocks of B (with the partner
 * dimension partitioned across threads via per-call coarse-N OMP).
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ETRMM_OMP_MIN 32

typedef long double T;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}
static int g_nb_trmm = 0;
static int trmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = env_int("ETRMM_NB", 64);
    return g_nb_trmm;
}

extern void egemm_(
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

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range scalar cores ─────────────────────────
 * Each core processes a column slice [j_start, j_end) of B with the
 * Netlib reference algorithm. Used both standalone (wrapped by the
 * OMP coarse-N macro) and inside the blocked path's diagonal-block
 * step (called on the ib×ib diagonal sub-matrix). */

static inline void trmm_lln_core(int j_start, int j_end, int M, T alpha,
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

static inline void trmm_lun_core(int j_start, int j_end, int M, T alpha,
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

static inline void trmm_llt_core(int j_start, int j_end, int M, T alpha,
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

static inline void trmm_lut_core(int j_start, int j_end, int M, T alpha,
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

/* ── SIDE = 'R' row-range scalar cores ──────────────────────────── */

static inline void trmm_rln_core(int i_start, int i_end, int N, T alpha,
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

static inline void trmm_run_core(int i_start, int i_end, int N, T alpha,
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

static inline void trmm_rlt_core(int i_start, int i_end, int N, T alpha,
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

static inline void trmm_rut_core(int i_start, int i_end, int N, T alpha,
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

/* ── Standalone OMP wrappers for the unblocked path. */

#ifdef _OPENMP
#define ETRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (N >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {             \
            _Pragma("omp parallel") {                                      \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                int js  = (int)((long long)N * tid / nt);                  \
                int je  = (int)((long long)N * (tid + 1) / nt);            \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }           \
    }
#define ETRMM_OMP_WRAP_R(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (M >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {             \
            _Pragma("omp parallel") {                                      \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                int is  = (int)((long long)M * tid / nt);                  \
                int ie  = (int)((long long)M * (tid + 1) / nt);            \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }           \
    }
#else
#define ETRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                      \
    }
#define ETRMM_OMP_WRAP_R(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                      \
    }
#endif

ETRMM_OMP_WRAP_L(trmm_lln, trmm_lln_core)
ETRMM_OMP_WRAP_L(trmm_lun, trmm_lun_core)
ETRMM_OMP_WRAP_L(trmm_llt, trmm_llt_core)
ETRMM_OMP_WRAP_L(trmm_lut, trmm_lut_core)
ETRMM_OMP_WRAP_R(trmm_rln, trmm_rln_core)
ETRMM_OMP_WRAP_R(trmm_run, trmm_run_core)
ETRMM_OMP_WRAP_R(trmm_rlt, trmm_rlt_core)
ETRMM_OMP_WRAP_R(trmm_rut, trmm_rut_core)

/* ── Blocked SIDE='L' variants ────────────────────────────────────
 *
 * For each variant, iterate M in nb-sized blocks. Block order is
 * picked so the trailing egemm reads only still-untouched portions
 * of B.
 *
 * Per block:
 *   - Run the unblocked diagonal scalar core on the ib×ib diagonal
 *     submatrix (in place on B_block); pass alpha through to the
 *     core, then do the trailing gemm with alpha to scale.
 *   - egemm call adds alpha · A_offdiag · B_offdiag into B_block
 *     with beta=1 (single dispatch per block; egemm internally runs
 *     coarse-N if its own threshold is met).
 */

enum trmm_variant { LLN, LUN, LLT, LUT };

/* Per-thread serial blocked-TRMM on a column slice [j_start, j_end) of B. */
static void blocked_chunk_L(enum trmm_variant V, int j_start, int j_end,
                            int M, int nb, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;

    const T one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    T *B_chunk = &B_(0, j_start);

    if (V == LLN) {
        /* B := alpha · L · B. Lower L: B_i depends on B_j for j ≤ i.
         * Iterate ic from largest down to 0 so B[0..ic) stays
         * untouched while we update B[ic..ic+ib). */
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            /* Diagonal block (in place, scaled by alpha): */
            trmm_lln_core(j_start, j_end, ib, alpha,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            /* Trailing gemm: B[ic..ic+ib, :] += alpha · L[ic..ic+ib, 0..ic] · B[0..ic, :] */
            if (ic > 0) {
                egemm_(NN, NN, &ib, &my_N, &ic, &alpha,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    } else if (V == LUN) {
        /* B := alpha · U · B. Upper U: B_i depends on B_j for j ≥ i.
         * Iterate ic from 0 upward. */
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            trmm_lun_core(j_start, j_end, ib, alpha,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                egemm_(NN, NN, &ib, &my_N, &trailing, &alpha,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else if (V == LLT) {
        /* B := alpha · Lᵀ · B. Lᵀ upper: B_i_new = alpha · Σ_{j≥i} L[j,i] B_j.
         * B_i depends on B_j for j ≥ i — iterate ic from 0 up. */
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            trmm_llt_core(j_start, j_end, ib, alpha,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                /* B[ic..ic+ib] += alpha · A[i0.., ic..ic+ib]ᵀ · B[i0.., :] */
                egemm_(TN, NN, &ib, &my_N, &trailing, &alpha,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else { /* LUT: B := alpha · Uᵀ · B. Iterate from largest ic down. */
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            trmm_lut_core(j_start, j_end, ib, alpha,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                /* B[ic..ic+ib] += alpha · A[0..ic, ic..ic+ib]ᵀ · B[0..ic, :] */
                egemm_(TN, NN, &ib, &my_N, &ic, &alpha,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    }
}

static void blocked_dispatch_L(enum trmm_variant V, int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trmm_nb();
#ifdef _OPENMP
    if (N >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = ((long long)N * tid) / nt;
            int je  = ((long long)N * (tid + 1)) / nt;
            blocked_chunk_L(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Blocked SIDE='R' variants ────────────────────────────────────
 * Symmetric to SIDE='L': iterate columns of B in nb-blocks. For each
 * column block, the diagonal trmm runs in-place; the egemm reads
 * a still-untouched column slice of B.
 *
 * The whole call is wrapped in one omp-parallel region so each
 * thread owns a row slice of B (partitioned along i ∈ [0, M)).
 * That row partition is independent across threads for all R-side
 * variants (every operation works column-by-column, with no
 * cross-row coupling).
 */

enum trmm_variant_R { RLN, RUN, RLT, RUT };

static void blocked_chunk_R(enum trmm_variant_R V, int i_start, int i_end,
                            int N, int nb, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    const T one = 1.0L;
    const char NN[1] = {'N'};
    const char NT[1] = {'N','T'};   /* unused literal; keep alignment */
    (void)NT;
    const char N_[1] = {'N'};
    const char T_[1] = {'T'};
    const int my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);   /* row stripe of B */
    /* Note: A(*,*) accessor uses original lda; B_chunk strides by
     * ldb columns and is offset by i_start rows. */

    if (V == RLN) {
        /* B := alpha · B · L. Lower L: col j depends on cols ≥ j.
         * Iterate jc from 0 up so cols [jc+jb..N) stay untouched. */
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            trmm_rln_core(i_start, i_end, jb, alpha,
                          &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                /* B[:, jc..jc+jb] += alpha · B[:, k0..N] · A[k0..N, jc..jc+jb] */
                egemm_(N_, N_, &my_M, &jb, &trailing, &alpha,
                       &B_chunk[(size_t)k0 * ldb], &ldb,
                       &A_(k0, jc), &lda, &one,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    } else if (V == RUN) {
        /* B := alpha · B · U. Upper U: col j depends on cols ≤ j.
         * Iterate jc from largest down. */
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            trmm_run_core(i_start, i_end, jb, alpha,
                          &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                /* B[:, jc..jc+jb] += alpha · B[:, 0..jc] · A[0..jc, jc..jc+jb] */
                egemm_(N_, N_, &my_M, &jb, &jc, &alpha,
                       B_chunk, &ldb,
                       &A_(0, jc), &lda, &one,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else if (V == RLT) {
        /* B := alpha · B · Lᵀ. Lᵀ upper: col j of result depends on
         * cols 0..j of B. Iterate jc from largest down. */
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            trmm_rlt_core(i_start, i_end, jb, alpha,
                          &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                /* B[:, jc..jc+jb] += alpha · B[:, 0..jc] · Lᵀ[0..jc, jc..jc+jb]
                 *                 = alpha · B[:, 0..jc] · L[jc..jc+jb, 0..jc]ᵀ */
                egemm_(N_, T_, &my_M, &jb, &jc, &alpha,
                       B_chunk, &ldb,
                       &A_(jc, 0), &lda, &one,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else { /* RUT: B := alpha · B · Uᵀ. Iterate jc from 0 up. */
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            trmm_rut_core(i_start, i_end, jb, alpha,
                          &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                /* B[:, jc..jc+jb] += alpha · B[:, k0..N] · Uᵀ[k0..N, jc..jc+jb]
                 *                 = alpha · B[:, k0..N] · U[jc..jc+jb, k0..N]ᵀ */
                egemm_(N_, T_, &my_M, &jb, &trailing, &alpha,
                       &B_chunk[(size_t)k0 * ldb], &ldb,
                       &A_(jc, k0), &lda, &one,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    }
}

static void blocked_dispatch_R(enum trmm_variant_R V, int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trmm_nb();
#ifdef _OPENMP
    if (M >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int is  = ((long long)M * tid) / nt;
            int ie  = ((long long)M * (tid + 1)) / nt;
            blocked_chunk_R(V, is, ie, N, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Entry point ──────────────────────────────────────────────── */

void etrmm_(
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
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0L) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0L;
        return;
    }

    const int nb = trmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(LLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(LUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(LLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(LUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        const int use_blocked = (N >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(RLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(RUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_run(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(RLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(RUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_rut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    }
}

#undef A_
#undef B_
