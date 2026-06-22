/* epblas_facade.h — per-shape macros that emit the two Fortran-ABI facades
 * (`name_` LP64 + `name_64_` ILP64) around one shared `ptrdiff_t` core.
 *
 * Design: see workspace/task/01-ilp64-interface-20260621/01-plan.md §2.
 *
 * The facade is the ONLY place the Fortran by-ref ABI is unwrapped:
 *   - integers/sizes/indices  -> deref'd to `ptrdiff_t` BY VALUE
 *   - flag chars              -> deref'd to `char` BY VALUE; hidden `*_len` dropped
 *   - alpha/beta (T or R)     -> the incoming `const T*`/`const R*` is FORWARDED
 *                                (>=16 B; by-value would spill a wide core — doc 02c)
 *   - array data + outputs    -> pointers forwarded unchanged
 *   - returns (T/R/index)     -> by value (i*amax casts ptrdiff_t -> int/int64_t)
 *
 * The two facades of a pair differ ONLY in the integer pointer width, so each
 * shape emits both from one `*_ONE(...,INT,SUF)` helper invoked twice (suffix
 * paste `name_` / `name_64_`): the twins are identical BY CONSTRUCTION. `i*amax`
 * (whose return type itself widens) is the sole hand-paired shape.
 *
 * Naming convention for the macro type parameters:
 *   T  = element/scalar type (real or complex)      VT = vector/array element
 *   ST = scalar type of alpha (may be the real type) RET = return type
 *   SA/SB = alpha/beta scalar types (her2k: alpha=T, beta=R)
 *
 * Hidden Fortran string-length args are declared (so the call-site ABI matches)
 * but unused — same as the pre-refactor entries, which also ignored them.
 *
 * `name_core(...)` must be declared/defined (external linkage if cross-called,
 * else `static`) in the including .c BEFORE the macro is invoked. Array/output
 * pointers keep their `restrict` in the core signature; the facade forwards
 * plain pointers (restrict is a callee hint, not part of the ABI).
 *
 * ── Type discipline (repo-wide) ───────────────────────────────────────────
 * `int` is a BOUNDARY-ONLY type. The fixed-width Fortran-ABI integers
 * (`int` LP64 / `int64_t` ILP64) appear in exactly two places, both in common/:
 *   (1) here — the facade layer, the public Fortran by-reference ABI; and
 *   (2) common/blas_omp.h — the OpenMP runtime wrapper, where the raw `int`
 *       from omp_get_max_threads()/omp_get_thread_num() is widened to ptrdiff_t
 *       at the boundary so no caller downstream handles `int`.
 * Inside the library (kind10/ kind16/ multifloats/) every quantity is one of:
 *   - ptrdiff_t — sizes, indices, leading dims, strides, thread ids/counts,
 *                 block sizes (the core's integer currency);
 *   - bool      — logical flags that are always 0/1 (nounit, upper, conj, …);
 *   - char      — BLAS option letters ('U'/'L'/'N'/'T'/'C'/'R'), normalised
 *                 once via blas_up() (common/blas_char.h) and compared as chars.
 * The guard scripts/check_int_boundary.py (ctest epblas_parallel_int_boundary_
 * guard) fails the suite if a bare `int` reappears in the three kind trees.
 */
#ifndef EPBLAS_FACADE_H
#define EPBLAS_FACADE_H

#include <stddef.h>   /* ptrdiff_t, size_t */
#include <stdint.h>   /* int64_t           */

#define EPBLAS_FACADE_PAIR(MAC, ...)                 \
    MAC(__VA_ARGS__, int,     _)                     \
    MAC(__VA_ARGS__, int64_t, _64_)

/* ============================ Level 1 ================================= */

/* SCAL: X := alpha*X.  escal(T,T); yscal(T,T); yescal real-alpha(R,T). */
#define EPBLAS_FACADE_SCAL_ONE(NAME, ST, VT, INT, SUF)                          \
    void NAME##SUF(const INT *n, const ST *alpha, VT *x, const INT *incx)       \
    { NAME##_core(*n, alpha, x, *incx); }
#define EPBLAS_FACADE_SCAL(NAME, ST, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SCAL_ONE, NAME, ST, VT)

/* ASUM / NRM2: RET f(n,x,incx).  easum(T,T) enrm2(T,T) eyasum(R,T) eynrm2(R,T). */
#define EPBLAS_FACADE_ASUM_ONE(NAME, RET, VT, INT, SUF)                         \
    RET NAME##SUF(const INT *n, const VT *x, const INT *incx)                   \
    { return NAME##_core(*n, x, *incx); }
#define EPBLAS_FACADE_ASUM(NAME, RET, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_ASUM_ONE, NAME, RET, VT)

/* AXPY: Y := alpha*X + Y. */
#define EPBLAS_FACADE_AXPY_ONE(NAME, ST, VT, INT, SUF)                          \
    void NAME##SUF(const INT *n, const ST *alpha, const VT *x, const INT *incx, \
                   VT *y, const INT *incy)                                      \
    { NAME##_core(*n, alpha, x, *incx, y, *incy); }
#define EPBLAS_FACADE_AXPY(NAME, ST, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_AXPY_ONE, NAME, ST, VT)

/* COPY: Y := X. */
#define EPBLAS_FACADE_COPY_ONE(NAME, VT, INT, SUF)                              \
    void NAME##SUF(const INT *n, const VT *x, const INT *incx,                  \
                   VT *y, const INT *incy)                                      \
    { NAME##_core(*n, x, *incx, y, *incy); }
#define EPBLAS_FACADE_COPY(NAME, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_COPY_ONE, NAME, VT)

/* SWAP: X <-> Y (x non-const). */
#define EPBLAS_FACADE_SWAP_ONE(NAME, VT, INT, SUF)                              \
    void NAME##SUF(const INT *n, VT *x, const INT *incx,                        \
                   VT *y, const INT *incy)                                      \
    { NAME##_core(*n, x, *incx, y, *incy); }
#define EPBLAS_FACADE_SWAP(NAME, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SWAP_ONE, NAME, VT)

/* DOT: RET f(n,x,incx,y,incy).  edot ydotu ydotc. */
#define EPBLAS_FACADE_DOT_ONE(NAME, RET, VT, INT, SUF)                          \
    RET NAME##SUF(const INT *n, const VT *x, const INT *incx,                   \
                  const VT *y, const INT *incy)                                 \
    { return NAME##_core(*n, x, *incx, y, *incy); }
#define EPBLAS_FACADE_DOT(NAME, RET, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_DOT_ONE, NAME, RET, VT)

/* ROT: apply plane rotation.  erot(c,s:T); yerot(c,s:R). */
#define EPBLAS_FACADE_ROT_ONE(NAME, ST, VT, INT, SUF)                           \
    void NAME##SUF(const INT *n, VT *x, const INT *incx, VT *y, const INT *incy,\
                   const ST *c, const ST *s)                                    \
    { NAME##_core(*n, x, *incx, y, *incy, c, s); }
#define EPBLAS_FACADE_ROT(NAME, ST, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_ROT_ONE, NAME, ST, VT)

/* ROTM: modified Givens apply.  erotm. */
#define EPBLAS_FACADE_ROTM_ONE(NAME, VT, INT, SUF)                              \
    void NAME##SUF(const INT *n, VT *x, const INT *incx, VT *y, const INT *incy,\
                   const VT *dparam)                                            \
    { NAME##_core(*n, x, *incx, y, *incy, dparam); }
#define EPBLAS_FACADE_ROTM(NAME, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_ROTM_ONE, NAME, VT)

/* I?AMAX: 1-based argmax(|x|). Return type widens with the ABI -> hand-paired. */
#define EPBLAS_FACADE_IAMAX(NAME, VT)                                           \
    int NAME##_(const int *n, const VT *x, const int *incx)                     \
    { return (int)NAME##_core(*n, x, *incx); }                                  \
    int64_t NAME##_64_(const int64_t *n, const VT *x, const int64_t *incx)      \
    { return (int64_t)NAME##_core(*n, x, *incx); }

/* ============================ Level 2 ================================= */

/* GEMV: y := alpha*op(A)*x + beta*y. */
#define EPBLAS_FACADE_GEMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *trans, const INT *m, const INT *n,               \
        const T *alpha, const T *a, const INT *lda, const T *x, const INT *incx,\
        const T *beta, T *y, const INT *incy, size_t trans_len)                 \
    { NAME##_core(*trans, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy); }
#define EPBLAS_FACADE_GEMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_GEMV_ONE, NAME, T)

/* GBMV: banded GEMV (+kl,ku). */
#define EPBLAS_FACADE_GBMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *trans, const INT *m, const INT *n,               \
        const INT *kl, const INT *ku, const T *alpha, const T *a, const INT *lda,\
        const T *x, const INT *incx, const T *beta, T *y, const INT *incy,      \
        size_t trans_len)                                                       \
    { NAME##_core(*trans, *m, *n, *kl, *ku, alpha, a, *lda, x, *incx, beta, y, *incy); }
#define EPBLAS_FACADE_GBMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_GBMV_ONE, NAME, T)

/* SYMV / HEMV: uplo,n,alpha,a,lda,x,incx,beta,y,incy. */
#define EPBLAS_FACADE_SYMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const INT *n, const T *alpha,              \
        const T *a, const INT *lda, const T *x, const INT *incx,                \
        const T *beta, T *y, const INT *incy, size_t uplo_len)                  \
    { NAME##_core(*uplo, *n, alpha, a, *lda, x, *incx, beta, y, *incy); }
#define EPBLAS_FACADE_SYMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SYMV_ONE, NAME, T)

/* SBMV / HBMV: banded SYMV/HEMV (+k). */
#define EPBLAS_FACADE_SBMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const INT *n, const INT *k, const T *alpha,\
        const T *a, const INT *lda, const T *x, const INT *incx,                \
        const T *beta, T *y, const INT *incy, size_t uplo_len)                  \
    { NAME##_core(*uplo, *n, *k, alpha, a, *lda, x, *incx, beta, y, *incy); }
#define EPBLAS_FACADE_SBMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SBMV_ONE, NAME, T)

/* SPMV / HPMV: packed SYMV/HEMV (ap, no lda). */
#define EPBLAS_FACADE_SPMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const INT *n, const T *alpha,              \
        const T *ap, const T *x, const INT *incx,                              \
        const T *beta, T *y, const INT *incy, size_t uplo_len)                  \
    { NAME##_core(*uplo, *n, alpha, ap, x, *incx, beta, y, *incy); }
#define EPBLAS_FACADE_SPMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SPMV_ONE, NAME, T)

/* TRMV / TRSV: uplo,trans,diag,n,a,lda,x,incx. */
#define EPBLAS_FACADE_TRMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const char *trans, const char *diag,       \
        const INT *n, const T *a, const INT *lda, T *x, const INT *incx,        \
        size_t uplo_len, size_t trans_len, size_t diag_len)                     \
    { NAME##_core(*uplo, *trans, *diag, *n, a, *lda, x, *incx); }
#define EPBLAS_FACADE_TRMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_TRMV_ONE, NAME, T)

/* TBMV / TBSV: banded TRMV/TRSV (+k). */
#define EPBLAS_FACADE_TBMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const char *trans, const char *diag,       \
        const INT *n, const INT *k, const T *a, const INT *lda, T *x,           \
        const INT *incx, size_t uplo_len, size_t trans_len, size_t diag_len)    \
    { NAME##_core(*uplo, *trans, *diag, *n, *k, a, *lda, x, *incx); }
#define EPBLAS_FACADE_TBMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_TBMV_ONE, NAME, T)

/* TPMV / TPSV: packed TRMV/TRSV (ap, no lda). */
#define EPBLAS_FACADE_TPMV_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const char *trans, const char *diag,       \
        const INT *n, const T *ap, T *x, const INT *incx,                       \
        size_t uplo_len, size_t trans_len, size_t diag_len)                     \
    { NAME##_core(*uplo, *trans, *diag, *n, ap, x, *incx); }
#define EPBLAS_FACADE_TPMV(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_TPMV_ONE, NAME, T)

/* GER / GERU / GERC: A := alpha*x*y' + A. */
#define EPBLAS_FACADE_GER_ONE(NAME, T, INT, SUF)                                \
    void NAME##SUF(const INT *m, const INT *n, const T *alpha,                  \
        const T *x, const INT *incx, const T *y, const INT *incy,               \
        T *a, const INT *lda)                                                   \
    { NAME##_core(*m, *n, alpha, x, *incx, y, *incy, a, *lda); }
#define EPBLAS_FACADE_GER(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_GER_ONE, NAME, T)

/* SYR / HER: A := alpha*x*x' + A.  esyr(alpha T,x/a T); yher(alpha R,x/a T). */
#define EPBLAS_FACADE_SYR_ONE(NAME, ST, VT, INT, SUF)                           \
    void NAME##SUF(const char *uplo, const INT *n, const ST *alpha,             \
        const VT *x, const INT *incx, VT *a, const INT *lda, size_t uplo_len)   \
    { NAME##_core(*uplo, *n, alpha, x, *incx, a, *lda); }
#define EPBLAS_FACADE_SYR(NAME, ST, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SYR_ONE, NAME, ST, VT)

/* SPR / HPR: packed SYR/HER (ap, no lda). */
#define EPBLAS_FACADE_SPR_ONE(NAME, ST, VT, INT, SUF)                           \
    void NAME##SUF(const char *uplo, const INT *n, const ST *alpha,             \
        const VT *x, const INT *incx, VT *ap, size_t uplo_len)                  \
    { NAME##_core(*uplo, *n, alpha, x, *incx, ap); }
#define EPBLAS_FACADE_SPR(NAME, ST, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SPR_ONE, NAME, ST, VT)

/* SYR2 / HER2: A := alpha*x*y' + (conj)alpha*y*x' + A. */
#define EPBLAS_FACADE_SYR2_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const INT *n, const T *alpha,              \
        const T *x, const INT *incx, const T *y, const INT *incy,               \
        T *a, const INT *lda, size_t uplo_len)                                  \
    { NAME##_core(*uplo, *n, alpha, x, *incx, y, *incy, a, *lda); }
#define EPBLAS_FACADE_SYR2(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SYR2_ONE, NAME, T)

/* SPR2 / HPR2: packed SYR2/HER2 (ap, no lda). */
#define EPBLAS_FACADE_SPR2_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *uplo, const INT *n, const T *alpha,              \
        const T *x, const INT *incx, const T *y, const INT *incy,               \
        T *ap, size_t uplo_len)                                                 \
    { NAME##_core(*uplo, *n, alpha, x, *incx, y, *incy, ap); }
#define EPBLAS_FACADE_SPR2(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SPR2_ONE, NAME, T)

/* ============================ Level 3 ================================= */

/* GEMM: C := alpha*op(A)*op(B) + beta*C. */
#define EPBLAS_FACADE_GEMM_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *transa, const char *transb, const INT *m,        \
        const INT *n, const INT *k, const T *alpha, const T *a, const INT *lda, \
        const T *b, const INT *ldb, const T *beta, T *c, const INT *ldc,        \
        size_t transa_len, size_t transb_len)                                   \
    { NAME##_core(*transa, *transb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc); }
#define EPBLAS_FACADE_GEMM(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_GEMM_ONE, NAME, T)

/* GEMMTR: triangular-output GEMM (uplo + 2 trans). */
#define EPBLAS_FACADE_GEMMTR_ONE(NAME, T, INT, SUF)                             \
    void NAME##SUF(const char *uplo, const char *transa, const char *transb,    \
        const INT *n, const INT *k, const T *alpha, const T *a, const INT *lda, \
        const T *b, const INT *ldb, const T *beta, T *c, const INT *ldc,        \
        size_t uplo_len, size_t ta_len, size_t tb_len)                          \
    { NAME##_core(*uplo, *transa, *transb, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc); }
#define EPBLAS_FACADE_GEMMTR(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_GEMMTR_ONE, NAME, T)

/* SYMM / HEMM: side,uplo,m,n,alpha,a,lda,b,ldb,beta,c,ldc. */
#define EPBLAS_FACADE_SYMM_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *side, const char *uplo, const INT *m,            \
        const INT *n, const T *alpha, const T *a, const INT *lda,               \
        const T *b, const INT *ldb, const T *beta, T *c, const INT *ldc,        \
        size_t side_len, size_t uplo_len)                                       \
    { NAME##_core(*side, *uplo, *m, *n, alpha, a, *lda, b, *ldb, beta, c, *ldc); }
#define EPBLAS_FACADE_SYMM(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SYMM_ONE, NAME, T)

/* SYRK / HERK: uplo,trans,n,k,alpha,a,lda,beta,c,ldc.
 * esyrk(alpha/beta T); yherk(alpha/beta R, a/c T) -> ST=scalar, VT=matrix. */
#define EPBLAS_FACADE_SYRK_ONE(NAME, ST, VT, INT, SUF)                          \
    void NAME##SUF(const char *uplo, const char *trans, const INT *n,           \
        const INT *k, const ST *alpha, const VT *a, const INT *lda,             \
        const ST *beta, VT *c, const INT *ldc, size_t uplo_len, size_t trans_len)\
    { NAME##_core(*uplo, *trans, *n, *k, alpha, a, *lda, beta, c, *ldc); }
#define EPBLAS_FACADE_SYRK(NAME, ST, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SYRK_ONE, NAME, ST, VT)

/* SYR2K / HER2K: uplo,trans,n,k,alpha,a,lda,b,ldb,beta,c,ldc.
 * esyr2k(alpha/beta T); yher2k(alpha T(=TC), beta R(=TR)) -> SA=alpha, SB=beta. */
#define EPBLAS_FACADE_SYR2K_ONE(NAME, SA, SB, VT, INT, SUF)                     \
    void NAME##SUF(const char *uplo, const char *trans, const INT *n,           \
        const INT *k, const SA *alpha, const VT *a, const INT *lda,             \
        const VT *b, const INT *ldb, const SB *beta, VT *c, const INT *ldc,     \
        size_t uplo_len, size_t trans_len)                                      \
    { NAME##_core(*uplo, *trans, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc); }
#define EPBLAS_FACADE_SYR2K(NAME, SA, SB, VT) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_SYR2K_ONE, NAME, SA, SB, VT)

/* TRMM / TRSM: side,uplo,transa,diag,m,n,alpha,a,lda,b,ldb. */
#define EPBLAS_FACADE_TRMM_ONE(NAME, T, INT, SUF)                               \
    void NAME##SUF(const char *side, const char *uplo, const char *transa,      \
        const char *diag, const INT *m, const INT *n, const T *alpha,           \
        const T *a, const INT *lda, T *b, const INT *ldb,                       \
        size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)   \
    { NAME##_core(*side, *uplo, *transa, *diag, *m, *n, alpha, a, *lda, b, *ldb); }
#define EPBLAS_FACADE_TRMM(NAME, T) EPBLAS_FACADE_PAIR(EPBLAS_FACADE_TRMM_ONE, NAME, T)

#endif /* EPBLAS_FACADE_H */
