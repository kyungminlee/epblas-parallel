"""L3 BLAS emitters: gemm, symm/hemm, syrk/herk, syr2k/her2k, trmm/trsm, gemmtr."""
from .core import TypeInfo
from .scaffold import sizes_body

def emit_gemm(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)M * (double)N * (double)K' if is_c else '2.0 * (double)M * (double)N * (double)K'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const int *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const int *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t);

static void run_one(char ta, char tb, int M, int N, int K, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    const size_t MKelt = (size_t)M * (size_t)K;
    const size_t KNelt = (size_t)K * (size_t)N;
    const size_t MNelt = (size_t)M * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, MKelt);
    {T} *B  = PERF_ALLOC({T}, KNelt);
    {T} *C  = PERF_ALLOC({T}, MNelt);
    {T} *Ci = PERF_ALLOC({T}, MNelt);
    int lda = M, ldb = K, ldc = M;
    {fill}({T}, A,  MKelt, 2);
    {fill}({T}, B,  KNelt, 3);
    {fill}({T}, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, {T});

    for (int r = 0; r < warmup; ++r) {{
        {name}_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, {T});
        {name}_migrated_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, {T});
    PERF_TIME(t_subject, iters, {name}_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));

    double flops = {flops};
    char key[3] = {{ta, tb, 0}};
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{{sizes_body((64, 128, 256, 512), is_c)}}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char *pairs[] = {{ {','.join('"' + p + '"' for p in (['NN','TN','NT','TT','CN','NC'] if is_c else ['NN','TN','NT','TT']))} }};
    for (size_t p = 0; p < sizeof(pairs)/sizeof(pairs[0]); ++p)
        for (int i = 0; i < n; ++i)
            run_one(pairs[p][0], pairs[p][1], sizes[i], sizes[i], sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L1 function-return: dot, dotu, dotc, asum, asum_c, nrm2 -----------------


def emit_symm_hemm(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)M * (double)M * (double)N' if is_c else '2.0 * (double)M * (double)M * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t);

static void run_one(char side, char uplo, int M, int N, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    {T} *A  = PERF_ALLOC({T}, AAelt);
    {T} *B  = PERF_ALLOC({T}, MNelt);
    {T} *C  = PERF_ALLOC({T}, MNelt);
    {T} *Ci = PERF_ALLOC({T}, MNelt);
    int lda = Asz, ldb = M, ldc = M;
    {fill}({T}, A,  AAelt, 2);
    {fill}({T}, B,  MNelt, 3);
    {fill}({T}, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, {T});
        {name}_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, {T});
    PERF_TIME(t_subject, iters, {name}_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    double flops = {flops};
    char key[3] = {{side, uplo, 0}};
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{{sizes_body((64, 128, 256, 512), is_c)}}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char sides[] = {{'L', 'R'}};
    const char uplos[] = {{'U', 'L'}};
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
        for (int i = 0; i < n; ++i)
            run_one(sides[s], uplos[u], sizes[i], sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L3 syrk/herk (UPLO, TRANS, N, K) ----------------------------------------


def emit_syrk_herk(name: str, ti: TypeInfo, is_c: bool, is_h: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    Talpha = ti.real_T if is_h else T
    p7 = ti.real_lit_p7 if is_h else (ti.cmplx_lit_p7 if is_c else ti.real_lit_p7)
    p3 = ti.real_lit_p3 if is_h else (ti.cmplx_lit_p3 if is_c else ti.real_lit_p3)
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '4.0 * (double)N * (double)N * (double)K' if is_c else '1.0 * (double)N * (double)N * (double)K'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const int *, const int *,
    const {Talpha} *, const {T} *, const int *,
    const {Talpha} *, {T} *, const int *, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const int *, const int *,
    const {Talpha} *, const {T} *, const int *,
    const {Talpha} *, {T} *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {{
    {Talpha} alpha = {p7}, beta = {p3};
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldc = N;
    {T} *A  = PERF_ALLOC({T}, AAelt);
    {T} *C  = PERF_ALLOC({T}, NNelt);
    {T} *Ci = PERF_ALLOC({T}, NNelt);
    {fill}({T}, A,  AAelt, 2);
    {fill}({T}, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, NNelt, {T});
        {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, NNelt, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, {T});
    PERF_TIME(t_subject, iters, {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, NNelt, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1));
    double flops = {flops};
    char key[3] = {{uplo, trans, 0}};
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(C); free(Ci);
}}

static const int default_sizes[] = {{{sizes_body((64, 128, 256, 512), is_c)}}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char uplos[] = {{'U', 'L'}};
    const char transes[] = {{ {', '.join("'" + c + "'" for c in (['N','C'] if is_h else ['N','T']))} }};
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < 2; ++t)
        for (int i = 0; i < n; ++i)
            run_one(uplos[u], transes[t], sizes[i], sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L3 syr2k/her2k (UPLO, TRANS, N, K) --------------------------------------


def emit_syr2k_her2k(name: str, ti: TypeInfo, is_c: bool, is_h: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    Talpha = T                              # syr2k: same as matrix; her2k: complex
    Tbeta = ti.real_T if is_h else T        # her2k: real beta; syr2k: matrix-typed beta
    p7_alpha = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3_beta = ti.real_lit_p3 if is_h else (ti.cmplx_lit_p3 if is_c else ti.real_lit_p3)
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '8.0 * (double)N * (double)N * (double)K' if is_c else '2.0 * (double)N * (double)N * (double)K'
    transes_list = ['N', 'C'] if is_h else (['N', 'T'])
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const int *, const int *,
    const {Talpha} *, const {T} *, const int *,
    const {T} *, const int *,
    const {Tbeta} *, {T} *, const int *, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const int *, const int *,
    const {Talpha} *, const {T} *, const int *,
    const {T} *, const int *,
    const {Tbeta} *, {T} *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {{
    {Talpha} alpha = {p7_alpha};
    {Tbeta}  beta  = {p3_beta};
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldb = A_rows, ldc = N;
    {T} *A  = PERF_ALLOC({T}, AAelt);
    {T} *B  = PERF_ALLOC({T}, AAelt);
    {T} *C  = PERF_ALLOC({T}, NNelt);
    {T} *Ci = PERF_ALLOC({T}, NNelt);
    {fill}({T}, A,  AAelt, 2);
    {fill}({T}, B,  AAelt, 3);
    {fill}({T}, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, NNelt, {T});
        {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, NNelt, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, {T});
    PERF_TIME(t_subject, iters, {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, NNelt, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    double flops = {flops};
    char key[3] = {{uplo, trans, 0}};
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{{sizes_body((64, 128, 256, 512), is_c)}}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char uplos[] = {{'U', 'L'}};
    const char transes[] = {{ {', '.join("'" + c + "'" for c in transes_list)} }};
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < 2; ++t)
        for (int i = 0; i < n; ++i)
            run_one(uplos[u], transes[t], sizes[i], sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L3 trmm/trsm (SIDE, UPLO, TRANS, DIAG, M, N) ----------------------------


def emit_trmm_trsm(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '4.0 * (double)M * (double)N * (double)M' if is_c else '1.0 * (double)M * (double)N * (double)M'
    from_d = 'Tc_from_d' if is_c else 'Tr_from_d'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const char *, const char *,
    const int *, const int *, const {T} *,
    const {T} *, const int *, {T} *, const int *,
    size_t, size_t, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const char *, const char *,
    const int *, const int *, const {T} *,
    const {T} *, const int *, {T} *, const int *,
    size_t, size_t, size_t, size_t);

static void run_one(char side, char uplo, char trans, char diag,
                    int M, int N, int iters, int warmup) {{
    {T} alpha = {p7};
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    int lda = Asz, ldb = M;
    {T} *A  = PERF_ALLOC({T}, AAelt);
    {T} *B  = PERF_ALLOC({T}, MNelt);
    {T} *Bi = PERF_ALLOC({T}, MNelt);
    {fill}({T}, A,  AAelt, 2);
    /* diagonal dominance for trsm */
    for (int i = 0; i < Asz; ++i) A[(size_t)i * lda + i] = {from_d}((double)(Asz + 4));
    {fill}({T}, Bi, MNelt, 4);
    PERF_RESET(B, Bi, MNelt, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);          PERF_RESET(B, Bi, MNelt, {T});
        {name}_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1); PERF_RESET(B, Bi, MNelt, {T});
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(B, Bi, MNelt, {T}),
        {name}_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(B, Bi, MNelt, {T}),
        {name}_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1));
    double flops = {flops};
    char key[5] = {{side, uplo, trans, diag, 0}};
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(Bi);
}}

static const int default_sizes[] = {{{sizes_body((64, 128, 256, 512), is_c)}}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    /* Sample over (side, uplo, trans, diag) — diag=N/U so the unit-diag
     * branch of trmm/trsm is exercised; full grid omits no categorical. */
    const char sides[] = {{'L', 'R'}};
    const char uplos[] = {{'U', 'L'}};
    const char transes[] = {{ {', '.join("'" + c + "'" for c in (['N','T','C'] if is_c else ['N','T']))} }};
    const char diags[]   = {{ 'N', 'U' }};
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
      for (size_t t = 0; t < sizeof(transes); ++t)
        for (size_t d = 0; d < sizeof(diags); ++d)
          for (int i = 0; i < n; ++i)
              run_one(sides[s], uplos[u], transes[t], diags[d], sizes[i], sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L3 gemmtr (UPLO, TRANSA, TRANSB, N, K) ----------------------------------


def emit_gemmtr(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = 'PERF_FILL_C' if is_c else 'PERF_FILL_R'
    flops = '4.0 * (double)N * (double)N * (double)K' if is_c else '1.0 * (double)N * (double)N * (double)K'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const char *,
    const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *,
    size_t, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const char *,
    const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *,
    size_t, size_t, size_t);

static void run_one(char uplo, char ta, char tb, int N, int K, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    int Arows = (ta == 'N') ? N : K;
    int Acols = (ta == 'N') ? K : N;
    int Brows = (tb == 'N') ? K : N;
    int Bcols = (tb == 'N') ? N : K;
    const size_t ABelt = (size_t)Arows * (size_t)Acols;
    const size_t BBelt = (size_t)Brows * (size_t)Bcols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = Arows, ldb = Brows, ldc = N;
    {T} *A  = PERF_ALLOC({T}, ABelt);
    {T} *B  = PERF_ALLOC({T}, BBelt);
    {T} *C  = PERF_ALLOC({T}, NNelt);
    {T} *Ci = PERF_ALLOC({T}, NNelt);
    {fill}({T}, A,  ABelt, 2);
    {fill}({T}, B,  BBelt, 3);
    {fill}({T}, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, {T});
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);          PERF_RESET(C, Ci, NNelt, {T});
        {name}_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1); PERF_RESET(C, Ci, NNelt, {T});
    }}
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, {T});
    PERF_TIME(t_subject, iters, {name}_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1));
    PERF_RESET(C, Ci, NNelt, {T});
    PERF_TIME(t_mg,      iters, {name}_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1));
    double flops = {flops};
    char key[4] = {{uplo, ta, tb, 0}};
    PERF_EMIT("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{{sizes_body((64, 128, 256, 512), is_c)}}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char uplos[] = {{'U', 'L'}};
    /* Sample full (ta, tb) grid: N/T for real, N/T/C for complex.
     * Trans choice flips the inner walk over A and B; covering all
     * combinations stresses every code path the kernel may take. */
    const char *pairs[] = {{ {', '.join('"' + a + b + '"' for a in (['N','T','C'] if is_c else ['N','T']) for b in (['N','T','C'] if is_c else ['N','T']))} }};
    for (size_t u = 0; u < 2; ++u)
        for (size_t p = 0; p < sizeof(pairs)/sizeof(pairs[0]); ++p)
            for (int i = 0; i < n; ++i)
                run_one(uplos[u], pairs[p][0], pairs[p][1], sizes[i], sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L1 rot / crot_r (vector rotation) -----------------------------------------

