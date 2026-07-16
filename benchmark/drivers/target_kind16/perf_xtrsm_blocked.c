/* 5-way timing + fuzz harness for xtrsm (kind16 complex).
 *
 *   (1) overlay xtrsm_ at OMP=1
 *   (2) overlay xtrsm_ at OMP=4
 *   (3) overlay xtrsm_blocked_ at OMP=1
 *   (4) overlay xtrsm_blocked_ at OMP=4
 *   (5) migrated xtrsm_migrated_ at OMP=1
 *
 * Only SIDE='L' is exercised (blocked variant falls through to xtrsm_
 * on SIDE='R'). Square shapes M=N for now.
 */

#include "../perf_common.h"

#include <quadmath.h>
#include <math.h>
#include <omp.h>

#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif

typedef __float128 Q16;
typedef _Complex float __attribute__((mode(TC))) X16;
#define X16_FROM(re, im) ((X16)((Q16)(double)(re) + 1.0i * (Q16)(double)(im)))
static inline X16 Tc_from_d(double d) { return (X16)((Q16)d); }

BLAS_EXTERN void xtrsm_(const char *, const char *, const char *, const char *,
    const int *, const int *, const X16 *,
    const X16 *, const int *, X16 *, const int *,
    size_t, size_t, size_t, size_t);
BLAS_EXTERN void xtrsm_blocked_(const char *, const char *, const char *, const char *,
    const int *, const int *, const X16 *,
    const X16 *, const int *, X16 *, const int *,
    size_t, size_t, size_t, size_t);
BLAS_EXTERN void xtrsm_migrated_(const char *, const char *, const char *, const char *,
    const int *, const int *, const X16 *,
    const X16 *, const int *, X16 *, const int *,
    size_t, size_t, size_t, size_t);

typedef void (*xtrsm_fn)(const char *, const char *, const char *, const char *,
    const int *, const int *, const X16 *,
    const X16 *, const int *, X16 *, const int *,
    size_t, size_t, size_t, size_t);

static Q16 q16_abs(Q16 z) { return z < 0 ? -z : z; }
static Q16 cabsq16(X16 z) {
    Q16 r = __real__ z, i = __imag__ z;
    Q16 ar = q16_abs(r), ai = q16_abs(i);
    if (ar < ai) { Q16 t = ar; ar = ai; ai = t; }
    if (ar == 0) return 0;
    Q16 q = ai / ar;
    return ar * sqrtq(1.0Q + q * q);
}

static int fuzz_check(const X16 *B_ref, const X16 *B_tst, int M, int N,
                      Q16 tol, const char *label) {
    int bad = 0;
    Q16 maxrelerr = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            size_t idx = (size_t)j * M + i;
            X16 d = B_ref[idx] - B_tst[idx];
            Q16 ad = cabsq16(d);
            Q16 mag = cabsq16(B_ref[idx]);
            Q16 denom = (mag > 1) ? mag : (Q16)1.0Q;
            Q16 rel = ad / denom;
            if (rel > maxrelerr) maxrelerr = rel;
            if (rel > tol) bad++;
        }
    }
    if (bad > 0) {
        char buf[64];
        quadmath_snprintf(buf, sizeof(buf), "%.6Qg", maxrelerr);
        fprintf(stderr,
                "[FUZZ FAIL] %s: %d/%d elems exceed tol; max relerr=%s\n",
                label, bad, M * N, buf);
    }
    return bad;
}

static double time_kernel(xtrsm_fn fn, int nthreads,
                          char side, char uplo, char trans, char diag,
                          int M, int N, X16 alpha,
                          const X16 *A, int lda,
                          const X16 *Bi, X16 *B, int ldb,
                          int iters, int warmup) {
    omp_set_num_threads(nthreads);
    size_t b_bytes = (size_t)M * (size_t)N * sizeof(X16);
    for (int r = 0; r < warmup; ++r) {
        memcpy(B, Bi, b_bytes);
        fn(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
    }
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        memcpy(B, Bi, b_bytes);
        fn(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
    }
    double t1 = perf_now_s();
    return (t1 - t0) / (iters ? iters : 1);
}

static void run_one(char uplo, char trans, char diag, int M, int N,
                    int iters, int warmup, Q16 tol) {
    char side = 'L';
    X16 alpha = X16_FROM(0.7, 0.0);
    int lda = M, ldb = M;
    size_t a_elems = (size_t)M * (size_t)M;
    size_t b_elems = (size_t)M * (size_t)N;

    X16 *A    = (X16 *)perf_aligned_alloc(64, a_elems * sizeof(X16));
    X16 *B    = (X16 *)perf_aligned_alloc(64, b_elems * sizeof(X16));
    X16 *Bi   = (X16 *)perf_aligned_alloc(64, b_elems * sizeof(X16));
    X16 *Bref = (X16 *)perf_aligned_alloc(64, b_elems * sizeof(X16));
    X16 *Btst = (X16 *)perf_aligned_alloc(64, b_elems * sizeof(X16));

    for (size_t i = 0; i < a_elems; ++i) {
        int s = 2;
        A[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131));
    }
    for (int i = 0; i < M; ++i) {
        A[(size_t)i * M + i] = Tc_from_d((double)(M + 4));
    }
    for (size_t i = 0; i < b_elems; ++i) {
        int s = 3;
        Bi[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131));
    }

    /* Correctness: blocked vs migrated. */
    memcpy(Bref, Bi, b_elems * sizeof(X16));
    omp_set_num_threads(1);
    xtrsm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha,
                    A, &lda, Bref, &ldb, 1, 1, 1, 1);
    memcpy(Btst, Bi, b_elems * sizeof(X16));
    omp_set_num_threads(4);
    xtrsm_blocked_(&side, &uplo, &trans, &diag, &M, &N, &alpha,
                   A, &lda, Btst, &ldb, 1, 1, 1, 1);
    char label[64];
    snprintf(label, sizeof(label), "xtrsm blk %c%c%c M=%d N=%d", uplo, trans, diag, M, N);
    fuzz_check(Bref, Btst, M, N, tol, label);

    double t_cur_1 = time_kernel(xtrsm_,         1, side, uplo, trans, diag, M, N, alpha,
                                  A, lda, Bi, B, ldb, iters, warmup);
    double t_cur_4 = time_kernel(xtrsm_,         4, side, uplo, trans, diag, M, N, alpha,
                                  A, lda, Bi, B, ldb, iters, warmup);
    double t_blk_1 = time_kernel(xtrsm_blocked_, 1, side, uplo, trans, diag, M, N, alpha,
                                  A, lda, Bi, B, ldb, iters, warmup);
    double t_blk_4 = time_kernel(xtrsm_blocked_, 4, side, uplo, trans, diag, M, N, alpha,
                                  A, lda, Bi, B, ldb, iters, warmup);
    double t_mig_1 = time_kernel(xtrsm_migrated_, 1, side, uplo, trans, diag, M, N, alpha,
                                  A, lda, Bi, B, ldb, iters, warmup);

    /* Bare wall time, ns/call (smaller = faster). */
    char key[16];
    snprintf(key, sizeof(key), "L%c%c%c", uplo, trans, diag);
    double ns_cur_1 = t_cur_1 * 1e9;
    double ns_cur_4 = t_cur_4 * 1e9;
    double ns_blk_1 = t_blk_1 * 1e9;
    double ns_blk_4 = t_blk_4 * 1e9;
    double ns_mig_1 = t_mig_1 * 1e9;

    /* Ratios are wall-time ratios (subject in numerator): < 1.0 = first faster.
     * cur4/cur1 = threading scaling of current; blk1/cur1, blk4/cur4 = blocked
     * vs current at matched OMP; blk4/mig1 = blocked-OMP4 vs migrated. */
    printf("%-10s  %-7s  %4dx%-4d %4d  %12.1f  %12.1f  %12.1f  %12.1f  %12.1f  "
           "%6.2fx %6.2fx %6.2fx %6.2fx\n",
           "xtrsm", key, M, N, iters,
           ns_cur_1, ns_cur_4, ns_blk_1, ns_blk_4, ns_mig_1,
           ns_cur_4 / ns_cur_1, ns_blk_1 / ns_cur_1,
           ns_blk_4 / ns_cur_4, ns_blk_4 / ns_mig_1);
    fflush(stdout);

    free(A); free(B); free(Bi); free(Bref); free(Btst);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  20);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 3);
    const char *tol_s = getenv("XTRSM_FUZZ_TOL");
    Q16 tol = (tol_s && *tol_s) ? strtoflt128(tol_s, NULL) : 1.0e-28Q;

    static const int default_sizes[] = {128, 256, 512};
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);

    printf("# columns: cur_1, cur_4, blk_1, blk_4, mig_1 (all in ns/call, smaller = faster)\n");
    printf("# ratios:  cur4/cur1  blk1/cur1  blk4/cur4  blk4/mig1  (wall ratios, < 1.0 = first faster)\n");

    /* nrhs-sweep mode: when BLAS_PERF_NRHS is set, sizes[] are M values,
     * nrhs comes from BLAS_PERF_NRHS, and we run only LUNN since the
     * parallel-scaling story is identical across UPLO×TR×DIAG (algorithm
     * touches the same data per (M, nrhs) regardless of which uplo/trans
     * branch xtrsm_blocked_ takes). */
    const char *nrhs_env = getenv("BLAS_PERF_NRHS");
    if (nrhs_env && *nrhs_env) {
        int nrhs_list[32];
        static const int nrhs_defaults[] = {1};
        int n_nrhs = perf_parse_int_list("BLAS_PERF_NRHS", nrhs_defaults,
            (int)(sizeof(nrhs_defaults)/sizeof(nrhs_defaults[0])), nrhs_list, 32);
        for (int i = 0; i < n; ++i) {
            int M = sizes[i];
            for (int k = 0; k < n_nrhs; ++k) {
                run_one('U', 'N', 'N', M, nrhs_list[k], iters, warmup, tol);
            }
        }
        return 0;
    }

    const char transes[] = { 'N', 'T', 'C' };
    const char diags[]   = { 'N', 'U' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
    for (size_t d = 0; d < sizeof(diags); ++d) {
        char uplo  = (u == 0) ? 'U' : 'L';
        char trans = transes[t];
        char diag  = diags[d];
        for (int i = 0; i < n; ++i) {
            int N = sizes[i];
            int M = sizes[i];
            run_one(uplo, trans, diag, M, N, iters, warmup, tol);
        }
    }
    return 0;
}
