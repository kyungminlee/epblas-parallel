"""L2 BLAS emitters for dense storage: gemv, ger, symv/hemv, syr/her, trmv/trsv."""
from .core import TypeInfo

def emit_gemv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)M * (double)N' if is_c else '2.0 * (double)M * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const int *, const {T} *,
    const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const int *, const {T} *,
    const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t);

static void run_one(char trans, int M, int N, int incx, int incy,
                    int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    const int XL = (trans == 'N') ? N : M;
    const int YL = (trans == 'N') ? M : N;
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(XL - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(YL - 1) * (size_t)absy;
    {T} *A  = PERF_ALLOC({T}, (size_t)M * N);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Y  = PERF_ALLOC({T}, leny);
    {T} *Yi = PERF_ALLOC({T}, leny);
    {fill}({T}, A,  (size_t)M * N, 2);
    {fill}({T}, X,  lenx, 3);
    {fill}({T}, Yi, leny, 4);

    PERF_RESET(Y, Yi, leny, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1);          PERF_RESET(Y, Yi, leny, {T});
        {name}_migrated_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1); PERF_RESET(Y, Yi, leny, {T});
    }}

    double t_subject, t_mg;
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_subject, iters, {name}_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1));
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1));

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

static const int default_sizes[] = {{128, 256, 512, 1024, 2048}};
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
    for (size_t t = 0; t < sizeof(transes); ++t) {{
        for (int xi = 0; xi < n_incx; ++xi) {{
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int yi = 0; yi < n_incy; ++yi) {{
                int incy = incys[yi]; if (incy == 0) continue;
                for (int i = 0; i < n; ++i)
                    run_one(transes[t], sizes[i], sizes[i], incx, incy, iters, warmup);
            }}
        }}
    }}
    return 0;
}}
'''

# -- L2 GER (and complex geru/gerc) ------------------------------------------


def emit_ger(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)M * (double)N' if is_c else '2.0 * (double)M * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const int *, const int *, const {T} *,
    const {T} *, const int *, const {T} *, const int *, {T} *, const int *);
BLAS_EXTERN void {name}_migrated_(const int *, const int *, const {T} *,
    const {T} *, const int *, const {T} *, const int *, {T} *, const int *);

static void run_one(int M, int N, int incx, int incy, int iters, int warmup) {{
    {T} alpha = {p7};
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(M - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    const size_t MNelt = (size_t)M * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, MNelt);
    {T} *Ai = PERF_ALLOC({T}, MNelt);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Y  = PERF_ALLOC({T}, leny);
    {fill}({T}, Ai, MNelt, 2);
    {fill}({T}, X,  lenx, 3);
    {fill}({T}, Y,  leny, 4);
    PERF_RESET(A, Ai, MNelt, {T});

    for (int r = 0; r < warmup; ++r) {{
        {name}_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);          PERF_RESET(A, Ai, MNelt, {T});
        {name}_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M); PERF_RESET(A, Ai, MNelt, {T});
    }}
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(A, Ai, MNelt, {T}), {name}_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(A, Ai, MNelt, {T}), {name}_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M));

    double flops = {flops};
    char key[24];
    if (incx == 1 && incy == 1) {{
        key[0] = '-'; key[1] = 0;
    }} else if (incy == 1) {{
        snprintf(key, sizeof(key), "x%d", incx);
    }} else if (incx == 1) {{
        snprintf(key, sizeof(key), "y%d", incy);
    }} else {{
        snprintf(key, sizeof(key), "x%d/y%d", incx, incy);
    }}
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(Ai); free(X); free(Y);
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
    for (int xi = 0; xi < n_incx; ++xi) {{
        int incx = incxs[xi]; if (incx == 0) continue;
        for (int yi = 0; yi < n_incy; ++yi) {{
            int incy = incys[yi]; if (incy == 0) continue;
            for (int i = 0; i < n; ++i) run_one(sizes[i], sizes[i], incx, incy, iters, warmup);
        }}
    }}
    return 0;
}}
'''

# -- L3 GEMM ------------------------------------------------------------------


def emit_symv_hemv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)N * (double)N' if is_c else '2.0 * (double)N * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const {T} *, const {T} *, const int *,
    const {T} *, const int *, const {T} *, {T} *, const int *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const {T} *, const {T} *, const int *,
    const {T} *, const int *, const {T} *, {T} *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int incy, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    const size_t NNelt = (size_t)N * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, NNelt);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Y  = PERF_ALLOC({T}, leny);
    {T} *Yi = PERF_ALLOC({T}, leny);
    {fill}({T}, A,  NNelt, 2);
    {fill}({T}, X,  lenx, 3);
    {fill}({T}, Yi, leny, 4);
    PERF_RESET(Y, Yi, leny, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1);          PERF_RESET(Y, Yi, leny, {T});
        {name}_migrated_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1); PERF_RESET(Y, Yi, leny, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_subject, iters, {name}_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1));
    PERF_RESET(Y, Yi, leny, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1));
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
                for (int i = 0; i < n; ++i) run_one(uplo, sizes[i], incx, incy, iters, warmup);
            }}
        }}
    }}
    return 0;
}}
'''

# -- L2 syr/her (UPLO, N) rank-1 ---------------------------------------------


def emit_syr_her(name: str, ti: TypeInfo, is_c: bool, is_her: bool) -> str:
    """syr/her: A := alpha*x*xT + A. For her, alpha is real."""
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.real_lit_p7 if is_her else (ti.cmplx_lit_p7 if is_c else ti.real_lit_p7)
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    Talpha = ti.real_T if is_her else T
    flops = '4.0 * (double)N * (double)N' if is_c else '1.0 * (double)N * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const {Talpha} *,
    const {T} *, const int *, {T} *, const int *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const {Talpha} *,
    const {T} *, const int *, {T} *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int iters, int warmup) {{
    {Talpha} alpha = {p7};
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t NNelt = (size_t)N * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, NNelt);
    {T} *Ai = PERF_ALLOC({T}, NNelt);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {fill}({T}, Ai, NNelt, 2);
    {fill}({T}, X,  lenx, 3);
    PERF_RESET(A, Ai, NNelt, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &alpha, X, &incx, A, &N, 1);          PERF_RESET(A, Ai, NNelt, {T});
        {name}_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1); PERF_RESET(A, Ai, NNelt, {T});
    }}
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(A, Ai, NNelt, {T}), {name}_(&uplo, &N, &alpha, X, &incx, A, &N, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(A, Ai, NNelt, {T}), {name}_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1));
    double flops = {flops};
    char key[16];
    if (incx == 1) {{
        key[0] = uplo; key[1] = 0;
    }} else {{
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }}
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(Ai); free(X);
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
    for (size_t u = 0; u < 2; ++u) {{
        char uplo = (u == 0) ? 'U' : 'L';
        for (int xi = 0; xi < n_incx; ++xi) {{
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, sizes[i], incx, iters, warmup);
        }}
    }}
    return 0;
}}
'''

# -- L2 spr/hpr (UPLO, N) packed rank-1 --------------------------------------


def emit_trmv_trsv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '4.0 * (double)N * (double)N' if is_c else '1.0 * (double)N * (double)N'
    from_d = 'Tc_from_d' if is_c else 'Tr_from_d'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const char *, const int *,
    const {T} *, const int *, {T} *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const char *, const int *,
    const {T} *, const int *, {T} *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int incx,
                    int iters, int warmup) {{
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t NNelt = (size_t)N * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, NNelt);
    {T} *X  = PERF_ALLOC({T}, lenx);
    {T} *Xi = PERF_ALLOC({T}, lenx);
    /* Diagonally dominant for trsv stability */
    {fill}({T}, A, NNelt, 2);
    for (int i = 0; i < N; ++i) {{
        size_t idx = (size_t)i * N + i;
        A[idx] = {from_d}((double)(N + 4));
    }}
    {fill}({T}, Xi, lenx, 3);
    PERF_RESET(X, Xi, lenx, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);          PERF_RESET(X, Xi, lenx, {T});
        {name}_migrated_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1); PERF_RESET(X, Xi, lenx, {T});
    }}
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, lenx, {T}), {name}_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, lenx, {T}), {name}_migrated_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1));
    double flops = {flops};
    /* Key encodes UPLO/TRANS/DIAG + stride. Examples: "LTN" (incx=1),
     * "LTN/x2" (incx=2), "LTN/x-1" (incx=-1). incx=1 keeps the old
     * key so existing reports stay parseable. */
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
            int incx = incxs[xi];
            if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, trans, diag, sizes[i], incx, iters, warmup);
        }}
    }}
    return 0;
}}
'''

# -- L2 tpmv/tpsv (UPLO, TRANS, DIAG, N) packed triangular -------------------

