"""L2 BLAS emitters for banded storage: gbmv, sbmv/hbmv, tbmv/tbsv."""
from .core import TypeInfo

def emit_gbmv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)(KL+KU+1) * (double)N' if is_c else '2.0 * (double)(KL+KU+1) * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const int *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const int *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t);

static void run_one(char trans, int M, int N, int KL, int KU,
                    int incx, int incy, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    int LDA = KL + KU + 1;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, Aelt);
    const int XL = (trans == 'N') ? N : M;
    const int YL = (trans == 'N') ? M : N;
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(XL - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(YL - 1) * (size_t)absy;
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Y  = PERF_ALLOC({T}, leny);
    {T} *Yi = PERF_ALLOC({T}, leny);
    {fill}({T}, A,  Aelt, 2);
    {fill}({T}, X,  lenx, 3);
    {fill}({T}, Yi, leny, 4);
    PERF_RESET(Y, Yi, leny, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);          PERF_RESET(Y, Yi, leny, {T});
        {name}_migrated_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1); PERF_RESET(Y, Yi, leny, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_subject, iters, {name}_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1));
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1));
    double flops = {flops};
    char key[24];
    if (incx == 1 && incy == 1) {{
        key[0] = trans; key[1] = 0;
    }} else if (incy == 1) {{
        snprintf(key, sizeof(key), "%c/x%d", trans, incx);
    }} else if (incx == 1) {{
        snprintf(key, sizeof(key), "%c/y%d", trans, incy);
    }} else {{
        snprintf(key, sizeof(key), "%c/x%d/y%d", trans, incx, incy);
    }}
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(X); free(Y); free(Yi);
}}

static const int default_sizes[] = {{128, 256, 512, 1024}};
static const int default_incxs[] = {{1, 2, -1}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8], incys[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    int n_incy = perf_parse_int_list("BLAS_PERF_INCY", incxs, n_incx, incys, 8);
    perf_print_header();
    const char transes[] = {{ {','.join("'" + c + "'" for c in (['N','T','C'] if is_c else ['N','T']))} }};
    for (size_t t = 0; t < sizeof(transes); ++t)
        for (int xi = 0; xi < n_incx; ++xi) {{
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int yi = 0; yi < n_incy; ++yi) {{
                int incy = incys[yi]; if (incy == 0) continue;
                for (int i = 0; i < n; ++i)
                    run_one(transes[t], sizes[i], sizes[i], 16, 16, incx, incy, iters, warmup);
            }}
        }}
    return 0;
}}
'''

# -- L2 banded: sbmv/hbmv (UPLO, N, K) ---------------------------------------


def emit_sbmv_hbmv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)(2*K+1) * (double)N' if is_c else '2.0 * (double)(2*K+1) * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t);

static void run_one(char uplo, int N, int K, int incx, int incy,
                    int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    int LDA = K + 1;
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, Aelt);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Y  = PERF_ALLOC({T}, leny);
    {T} *Yi = PERF_ALLOC({T}, leny);
    {fill}({T}, A,  Aelt, 2);
    {fill}({T}, X,  lenx, 3);
    {fill}({T}, Yi, leny, 4);
    PERF_RESET(Y, Yi, leny, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);          PERF_RESET(Y, Yi, leny, {T});
        {name}_migrated_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1); PERF_RESET(Y, Yi, leny, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_subject, iters, {name}_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1));
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1));
    double flops = {flops};
    char key[24];
    if (incx == 1 && incy == 1) {{
        key[0] = uplo; key[1] = 0;
    }} else if (incy == 1) {{
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }} else if (incx == 1) {{
        snprintf(key, sizeof(key), "%c/y%d", uplo, incy);
    }} else {{
        snprintf(key, sizeof(key), "%c/x%d/y%d", uplo, incx, incy);
    }}
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(X); free(Y); free(Yi);
}}

static const int default_sizes[] = {{128, 256, 512, 1024}};
static const int default_incxs[] = {{1, 2, -1}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8], incys[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    int n_incy = perf_parse_int_list("BLAS_PERF_INCY", incxs, n_incx, incys, 8);
    perf_print_header();
    for (size_t u = 0; u < 2; ++u) {{
        char uplo = (u == 0) ? 'U' : 'L';
        for (int xi = 0; xi < n_incx; ++xi) {{
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int yi = 0; yi < n_incy; ++yi) {{
                int incy = incys[yi]; if (incy == 0) continue;
                for (int i = 0; i < n; ++i) run_one(uplo, sizes[i], 16, incx, incy, iters, warmup);
            }}
        }}
    }}
    return 0;
}}
'''

# -- L2 banded triangular: tbmv/tbsv (UPLO, TRANS, DIAG, N, K) ---------------


def emit_tbmv_tbsv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '4.0 * (double)(2*K+1) * (double)N' if is_c else '1.0 * (double)(2*K+1) * (double)N'
    from_d = 'Tc_from_d' if is_c else 'Tr_from_d'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const char *, const int *, const int *,
    const {T} *, const int *, {T} *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const char *, const int *, const int *,
    const {T} *, const int *, {T} *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int K, int incx,
                    int iters, int warmup) {{
    int LDA = K + 1;
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, Aelt);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Xi = PERF_ALLOC({T}, lenx);
    {fill}({T}, A, Aelt, 2);
    /* diagonal at known row of band — large to stabilize tbsv */
    int diag_row = (uplo == 'U') ? K : 0;
    for (int j = 0; j < N; ++j) A[(size_t)j * LDA + diag_row] = {from_d}((double)(K + 4));
    {fill}({T}, Xi, lenx, 3);
    PERF_RESET(X, Xi, lenx, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);          PERF_RESET(X, Xi, lenx, {T});
        {name}_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1); PERF_RESET(X, Xi, lenx, {T});
    }}
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, lenx, {T}), {name}_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, lenx, {T}), {name}_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1));
    double flops = {flops};
    char key[16];
    if (incx == 1) {{
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    }} else {{
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }}
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(X); free(Xi);
}}

static const int default_sizes[] = {{128, 256, 512, 1024}};
static const int default_incxs[] = {{1, 2, -1}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    perf_print_header();
    const char transes[] = {{ {','.join("'" + c + "'" for c in (['N','T','C'] if is_c else ['N','T']))} }};
    const char diags[]   = {{ 'N', 'U' }};
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
    for (size_t d = 0; d < sizeof(diags); ++d) {{
        char uplo = (u == 0) ? 'U' : 'L';
        char trans = transes[t];
        char diag = diags[d];
        for (int xi = 0; xi < n_incx; ++xi) {{
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, trans, diag, sizes[i], 16, incx, iters, warmup);
        }}
    }}
    return 0;
}}
'''

# -- L3 symm/hemm (SIDE, UPLO, M, N) -----------------------------------------

