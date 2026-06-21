"""Per-shape Spec builders for the dual-link generator (dual.py).

Each builder returns (is_c, ret, sig, Spec). Types in `sig` use the driver's
`T` typedef (CT for complex, RT for real) and `RT` for forced-real scalars
(herk/her2k alpha-beta). The grid/setup mirror the legacy emit_*.py harnesses
so the dual driver exercises the same cells.

Currently implemented: L3 (gemm, symm/hemm, syrk/herk, syr2k/her2k, trmm/trsm,
gemmtr). L2/L1 shapes are added incrementally after the L3 pipeline validates
against the qsyrk/qsyr2k reps=40 ground truth.
"""
from __future__ import annotations

from .core import routine_shape
from .dual import Spec, fill, iters_for, family_for

# literal element constructors
RL7, RL3 = 'MKR(0.7)', 'MKR(0.3)'
CL7, CL3 = 'MKC(0.7, 0.0)', 'MKC(0.3, 0.0)'


def _size_iters(sizes, klass):
    its = [iters_for(s, klass) for s in sizes]
    return (f'    const int sizes[] = {{{", ".join(map(str, sizes))}}};\n'
            f'    const int itersarr[] = {{{", ".join(map(str, its))}}};\n'
            f'    const int nsz = (int)(sizeof(sizes) / sizeof(sizes[0]));')


def _chars(cs):
    return '{' + ', '.join(f"'{c}'" for c in cs) + '}'


def _l3_sizes(is_c):
    return (64, 128, 256) if is_c else (64, 128, 256, 512)


# ---------------------------------------------------------------------------
# gemm
# ---------------------------------------------------------------------------
def build_gemm(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const char *, const int *, const int *, const int *, '
           'const T *, const T *, const int *, const T *, const int *, '
           'const T *, T *, const int *, size_t, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    const size_t MKelt = (size_t)M * (size_t)K, KNelt = (size_t)K * (size_t)N, MNelt = (size_t)M * (size_t)N;
    int lda = M, ldb = K, ldc = M;
    T *A = (T *)aligned_alloc(64, MKelt * sizeof(T));
    T *B = (T *)aligned_alloc(64, KNelt * sizeof(T));
    T *C = (T *)aligned_alloc(64, MNelt * sizeof(T));
    T *Ci = (T *)aligned_alloc(64, MNelt * sizeof(T));
    {fill('A', 'MKelt', is_c, 0)}
    {fill('B', 'KNelt', is_c, 2)}
    {fill('Ci', 'MNelt', is_c, 4)}
    memcpy(C, Ci, MNelt * sizeof(T));
    char kbuf[4]; kbuf[0] = ta; kbuf[1] = tb; kbuf[2] = 0;'''
    pairs = ['NN', 'TN', 'NT', 'TT'] + (['CN', 'NC'] if is_c else [])
    sweep = f'''    const char *pairs[] = {{{", ".join('"' + p + '"' for p in pairs)}}};
{_size_iters(_l3_sizes(is_c), 'l3')}
    for (size_t p = 0; p < sizeof(pairs) / sizeof(pairs[0]); ++p)
        for (int i = 0; i < nsz; ++i)
            run_one(pairs[p][0], pairs[p][1], sizes[i], sizes[i], sizes[i], itersarr[i], reps);'''
    spec = Spec(params='char ta, char tb, int M, int N, int K', setup=setup,
                call='&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1',
                reset='memcpy(C, Ci, MNelt * sizeof(T));',
                touch='((const unsigned long *)(const void *)C)[0]',
                free='    free(A); free(B); free(C); free(Ci);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# symm / hemm
# ---------------------------------------------------------------------------
def build_symm_hemm(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const char *, const int *, const int *, '
           'const T *, const T *, const int *, const T *, const int *, '
           'const T *, T *, const int *, size_t, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz, MNelt = (size_t)M * (size_t)N;
    int lda = Asz, ldb = M, ldc = M;
    T *A = (T *)aligned_alloc(64, AAelt * sizeof(T));
    T *B = (T *)aligned_alloc(64, MNelt * sizeof(T));
    T *C = (T *)aligned_alloc(64, MNelt * sizeof(T));
    T *Ci = (T *)aligned_alloc(64, MNelt * sizeof(T));
    {fill('A', 'AAelt', is_c, 0)}
    {fill('B', 'MNelt', is_c, 2)}
    {fill('Ci', 'MNelt', is_c, 4)}
    memcpy(C, Ci, MNelt * sizeof(T));
    char kbuf[4]; kbuf[0] = side; kbuf[1] = uplo; kbuf[2] = 0;'''
    sweep = f'''    const char sides[] = {_chars(['L', 'R'])};
    const char uplos[] = {_chars(['U', 'L'])};
{_size_iters(_l3_sizes(is_c), 'l3')}
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
        for (int i = 0; i < nsz; ++i)
            run_one(sides[s], uplos[u], sizes[i], sizes[i], itersarr[i], reps);'''
    spec = Spec(params='char side, char uplo, int M, int N', setup=setup,
                call='&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1',
                reset='memcpy(C, Ci, MNelt * sizeof(T));',
                touch='((const unsigned long *)(const void *)C)[0]',
                free='    free(A); free(B); free(C); free(Ci);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# syrk / herk
# ---------------------------------------------------------------------------
def build_syrk_herk(name, is_c, is_h):
    if is_h:
        Ta, p7, p3 = 'RT', RL7, RL3
    elif is_c:
        Ta, p7, p3 = 'CT', CL7, CL3
    else:
        Ta, p7, p3 = 'RT', RL7, RL3
    sig = ('const char *, const char *, const int *, const int *, '
           f'const {Ta} *, const T *, const int *, '
           f'const {Ta} *, T *, const int *, size_t, size_t')
    setup = f'''    {Ta} alpha = {p7}, beta = {p3};
    int A_rows = (trans == 'N') ? N : K, A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols, NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldc = N;
    T *A = (T *)aligned_alloc(64, AAelt * sizeof(T));
    T *C = (T *)aligned_alloc(64, NNelt * sizeof(T));
    T *Ci = (T *)aligned_alloc(64, NNelt * sizeof(T));
    {fill('A', 'AAelt', is_c, 0)}
    {fill('Ci', 'NNelt', is_c, 4)}
    memcpy(C, Ci, NNelt * sizeof(T));
    char kbuf[4]; kbuf[0] = uplo; kbuf[1] = trans; kbuf[2] = 0;'''
    transes = ['N', 'C'] if is_h else ['N', 'T']
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
    const char transes[] = {_chars(transes)};
{_size_iters(_l3_sizes(is_c), 'l3')}
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < 2; ++t)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], transes[t], sizes[i], sizes[i], itersarr[i], reps);'''
    spec = Spec(params='char uplo, char trans, int N, int K', setup=setup,
                call='&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1',
                reset='memcpy(C, Ci, NNelt * sizeof(T));',
                touch='((const unsigned long *)(const void *)C)[0]',
                free='    free(A); free(C); free(Ci);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# syr2k / her2k
# ---------------------------------------------------------------------------
def build_syr2k_her2k(name, is_c, is_h):
    Ta = 'CT' if is_c else 'RT'
    p7a = CL7 if is_c else RL7
    Tb = 'RT' if is_h else ('CT' if is_c else 'RT')
    p3b = RL3 if is_h else (CL3 if is_c else RL3)
    sig = ('const char *, const char *, const int *, const int *, '
           f'const {Ta} *, const T *, const int *, const T *, const int *, '
           f'const {Tb} *, T *, const int *, size_t, size_t')
    setup = f'''    {Ta} alpha = {p7a};
    {Tb} beta = {p3b};
    int A_rows = (trans == 'N') ? N : K, A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols, NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldb = A_rows, ldc = N;
    T *A = (T *)aligned_alloc(64, AAelt * sizeof(T));
    T *B = (T *)aligned_alloc(64, AAelt * sizeof(T));
    T *C = (T *)aligned_alloc(64, NNelt * sizeof(T));
    T *Ci = (T *)aligned_alloc(64, NNelt * sizeof(T));
    {fill('A', 'AAelt', is_c, 0)}
    {fill('B', 'AAelt', is_c, 2)}
    {fill('Ci', 'NNelt', is_c, 4)}
    memcpy(C, Ci, NNelt * sizeof(T));
    char kbuf[4]; kbuf[0] = uplo; kbuf[1] = trans; kbuf[2] = 0;'''
    transes = ['N', 'C'] if is_h else ['N', 'T']
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
    const char transes[] = {_chars(transes)};
{_size_iters(_l3_sizes(is_c), 'l3')}
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < 2; ++t)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], transes[t], sizes[i], sizes[i], itersarr[i], reps);'''
    spec = Spec(params='char uplo, char trans, int N, int K', setup=setup,
                call='&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1',
                reset='memcpy(C, Ci, NNelt * sizeof(T));',
                touch='((const unsigned long *)(const void *)C)[0]',
                free='    free(A); free(B); free(C); free(Ci);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# trmm / trsm
# ---------------------------------------------------------------------------
def build_trmm_trsm(name, is_c):
    p7 = CL7 if is_c else RL7
    dom = 'MKC((double)(Asz + 4), 0.0)' if is_c else 'MKR((double)(Asz + 4))'
    sig = ('const char *, const char *, const char *, const char *, '
           'const int *, const int *, const T *, const T *, const int *, T *, const int *, '
           'size_t, size_t, size_t, size_t')
    setup = f'''    T alpha = {p7};
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz, MNelt = (size_t)M * (size_t)N;
    int lda = Asz, ldb = M;
    T *A = (T *)aligned_alloc(64, AAelt * sizeof(T));
    T *B = (T *)aligned_alloc(64, MNelt * sizeof(T));
    T *Bi = (T *)aligned_alloc(64, MNelt * sizeof(T));
    {fill('A', 'AAelt', is_c, 0)}
    for (int d = 0; d < Asz; ++d) A[(size_t)d * lda + d] = {dom};
    {fill('Bi', 'MNelt', is_c, 4)}
    memcpy(B, Bi, MNelt * sizeof(T));
    char kbuf[6]; kbuf[0] = side; kbuf[1] = uplo; kbuf[2] = trans; kbuf[3] = diag; kbuf[4] = 0;'''
    transes = ['N', 'T', 'C'] if is_c else ['N', 'T']
    sweep = f'''    const char sides[] = {_chars(['L', 'R'])};
    const char uplos[] = {_chars(['U', 'L'])};
    const char transes[] = {_chars(transes)};
    const char diags[] = {_chars(['N', 'U'])};
{_size_iters(_l3_sizes(is_c), 'l3')}
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
      for (size_t t = 0; t < sizeof(transes); ++t) for (size_t d = 0; d < sizeof(diags); ++d)
        for (int i = 0; i < nsz; ++i)
            run_one(sides[s], uplos[u], transes[t], diags[d], sizes[i], sizes[i], itersarr[i], reps);'''
    spec = Spec(params='char side, char uplo, char trans, char diag, int M, int N', setup=setup,
                call='&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1',
                reset='memcpy(B, Bi, MNelt * sizeof(T));',
                touch='((const unsigned long *)(const void *)B)[0]',
                free='    free(A); free(B); free(Bi);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# gemmtr
# ---------------------------------------------------------------------------
def build_gemmtr(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const char *, const char *, const int *, const int *, '
           'const T *, const T *, const int *, const T *, const int *, '
           'const T *, T *, const int *, size_t, size_t, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int Arows = (ta == 'N') ? N : K, Acols = (ta == 'N') ? K : N;
    int Brows = (tb == 'N') ? K : N, Bcols = (tb == 'N') ? N : K;
    const size_t ABelt = (size_t)Arows * (size_t)Acols, BBelt = (size_t)Brows * (size_t)Bcols, NNelt = (size_t)N * (size_t)N;
    int lda = Arows, ldb = Brows, ldc = N;
    T *A = (T *)aligned_alloc(64, ABelt * sizeof(T));
    T *B = (T *)aligned_alloc(64, BBelt * sizeof(T));
    T *C = (T *)aligned_alloc(64, NNelt * sizeof(T));
    T *Ci = (T *)aligned_alloc(64, NNelt * sizeof(T));
    {fill('A', 'ABelt', is_c, 0)}
    {fill('B', 'BBelt', is_c, 2)}
    {fill('Ci', 'NNelt', is_c, 4)}
    memcpy(C, Ci, NNelt * sizeof(T));
    char kbuf[4]; kbuf[0] = uplo; kbuf[1] = ta; kbuf[2] = tb; kbuf[3] = 0;'''
    tcs = ['N', 'T', 'C'] if is_c else ['N', 'T']
    pairs = [a + b for a in tcs for b in tcs]
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
    const char *pairs[] = {{{", ".join('"' + p + '"' for p in pairs)}}};
{_size_iters(_l3_sizes(is_c), 'l3')}
    for (size_t u = 0; u < 2; ++u)
        for (size_t p = 0; p < sizeof(pairs) / sizeof(pairs[0]); ++p)
            for (int i = 0; i < nsz; ++i)
                run_one(uplos[u], pairs[p][0], pairs[p][1], sizes[i], sizes[i], itersarr[i], reps);'''
    spec = Spec(params='char uplo, char ta, char tb, int N, int K', setup=setup,
                call='&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1',
                reset='memcpy(C, Ci, NNelt * sizeof(T));',
                touch='((const unsigned long *)(const void *)C)[0]',
                free='    free(A); free(B); free(C); free(Ci);', sweep=sweep)
    return is_c, 'void', sig, spec


# ===========================================================================
# L2 / L1 shapes
# ===========================================================================
# Stride sweep: unit, positive-strided, negative-strided. Both incx and incy
# (where present) take the same value per cell — the diagonal of the legacy
# full incx*incy cross, which captures unit/strided/reversed codegen without
# the 9x cell blow-up that fp128/dd can't afford in a 3-leg dual run.
_STRIDES = (1, 2, -1)
_ALLOC = '(T *)aligned_alloc(64, '


def _strides_decl():
    return (f'    const int incs[] = {{{", ".join(map(str, _STRIDES))}}};\n'
            '    const int ninc = (int)(sizeof(incs) / sizeof(incs[0]));')


def _l2_sizes(is_c):
    return (128, 256, 512) if is_c else (128, 256, 512, 1024)


def _l1_sizes(is_c):
    # span cache-resident -> past-L3 so the RMW threading crossover is visible.
    # The harness holds element-ops constant per cell (budget / N), so every cell
    # costs ~equal wall time EXCEPT the 1 M point, which thrashes memory — and a
    # complex DD element is 32 B (vs 16 B real), so the scalar mig leg streams the
    # 32 MB complex buffer ~2-4x slower, making that one cell dominate the whole
    # complex L1 bench. Drop it for complex (the 64 Ki cell already shows the
    # past-L2 crossover); the real side keeps the full past-L3 span.
    return (1024, 65536) if is_c else (1024, 65536, 1048576)


# kbuf builders (set in setup): keep the unit-stride key bare so it matches the
# legacy reports; suffix "/x<inc>" only when strided.
def _kbuf_inc(chars_expr, inc='inc'):
    """chars_expr: list of single-char C expressions forming the leading key."""
    n = len(chars_expr)
    assign = '; '.join(f'kbuf[{i}] = {c}' for i, c in enumerate(chars_expr))
    fmt = '%c' * n
    args = ', '.join(chars_expr)
    return (f'''    char kbuf[16];
    if ({inc} == 1) {{ {assign}; kbuf[{n}] = 0; }}
    else snprintf(kbuf, sizeof(kbuf), "{fmt}/x%d", {args}, {inc});''')


def _out_touch(buf):
    return f'((const unsigned long *)(const void *)({buf}))[0]'


# ---------------------------------------------------------------------------
# L2 dense
# ---------------------------------------------------------------------------
def build_gemv(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const int *, const int *, const T *, const T *, '
           'const int *, const T *, const int *, const T *, T *, const int *, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int XL = (trans == 'N') ? N : M, YL = (trans == 'N') ? M : N;
    int absi = inc < 0 ? -inc : inc, lda = M;
    const size_t lenx = (size_t)1 + (size_t)(XL - 1) * (size_t)absi;
    const size_t leny = (size_t)1 + (size_t)(YL - 1) * (size_t)absi;
    const size_t MNelt = (size_t)M * (size_t)N;
    T *A = {_ALLOC}MNelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}leny * sizeof(T));
    T *Yi = {_ALLOC}leny * sizeof(T));
    {fill('A', 'MNelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Yi', 'leny', is_c, 4)}
    memcpy(Y, Yi, leny * sizeof(T));
{_kbuf_inc(['trans'])}'''
    transes = ['N', 'T', 'C'] if is_c else ['N', 'T']
    sweep = f'''    const char transes[] = {_chars(transes)};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t t = 0; t < sizeof(transes); ++t) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(transes[t], sizes[i], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char trans, int M, int N, int inc', setup=setup,
                call='&trans, &M, &N, &alpha, A, &lda, X, &inc, &beta, Y, &inc, 1',
                reset='memcpy(Y, Yi, leny * sizeof(T));', touch=_out_touch('Y'),
                free='    free(A); free(X); free(Y); free(Yi);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_ger(name, is_c):
    p7 = CL7 if is_c else RL7
    sig = ('const int *, const int *, const T *, const T *, const int *, '
           'const T *, const int *, T *, const int *')
    setup = f'''    T alpha = {p7};
    int absi = inc < 0 ? -inc : inc, lda = M;
    const size_t lenx = (size_t)1 + (size_t)(M - 1) * (size_t)absi;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t MNelt = (size_t)M * (size_t)N;
    T *A = {_ALLOC}MNelt * sizeof(T));
    T *Ai = {_ALLOC}MNelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}leny * sizeof(T));
    {fill('Ai', 'MNelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Y', 'leny', is_c, 4)}
    memcpy(A, Ai, MNelt * sizeof(T));
    char kbuf[16];
    if (inc == 1) {{ kbuf[0] = '-'; kbuf[1] = 0; }}
    else snprintf(kbuf, sizeof(kbuf), "x%d", inc);'''
    sweep = f'''{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(sizes[i], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='int M, int N, int inc', setup=setup,
                call='&M, &N, &alpha, X, &inc, Y, &inc, A, &lda',
                reset='memcpy(A, Ai, MNelt * sizeof(T));', touch=_out_touch('A'),
                free='    free(A); free(Ai); free(X); free(Y);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_symv_hemv(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const int *, const T *, const T *, const int *, '
           'const T *, const int *, const T *, T *, const int *, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int absi = inc < 0 ? -inc : inc, lda = N;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t NNelt = (size_t)N * (size_t)N;
    T *A = {_ALLOC}NNelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}leny * sizeof(T));
    T *Yi = {_ALLOC}leny * sizeof(T));
    {fill('A', 'NNelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Yi', 'leny', is_c, 4)}
    memcpy(Y, Yi, leny * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &alpha, A, &lda, X, &inc, &beta, Y, &inc, 1',
                reset='memcpy(Y, Yi, leny * sizeof(T));', touch=_out_touch('Y'),
                free='    free(A); free(X); free(Y); free(Yi);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_syr_her(name, is_c, is_h):
    Ta = 'RT' if is_h else ('CT' if is_c else 'RT')
    p7 = RL7 if is_h else (CL7 if is_c else RL7)
    sig = (f'const char *, const int *, const {Ta} *, const T *, const int *, '
           'T *, const int *, size_t')
    setup = f'''    {Ta} alpha = {p7};
    int absi = inc < 0 ? -inc : inc, lda = N;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t NNelt = (size_t)N * (size_t)N;
    T *A = {_ALLOC}NNelt * sizeof(T));
    T *Ai = {_ALLOC}NNelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    {fill('Ai', 'NNelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    memcpy(A, Ai, NNelt * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &alpha, X, &inc, A, &lda, 1',
                reset='memcpy(A, Ai, NNelt * sizeof(T));', touch=_out_touch('A'),
                free='    free(A); free(Ai); free(X);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_syr2_her2(name, is_c):
    """Dense symmetric/Hermitian rank-2 update (syr2/her2). alpha is the
    element type (complex for her2). x and y share the stride `inc`."""
    Ta = 'CT' if is_c else 'RT'
    p7 = CL7 if is_c else RL7
    sig = (f'const char *, const int *, const {Ta} *, const T *, const int *, '
           'const T *, const int *, T *, const int *, size_t')
    setup = f'''    {Ta} alpha = {p7};
    int absi = inc < 0 ? -inc : inc, lda = N;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t NNelt = (size_t)N * (size_t)N;
    T *A = {_ALLOC}NNelt * sizeof(T));
    T *Ai = {_ALLOC}NNelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}lenx * sizeof(T));
    {fill('Ai', 'NNelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Y', 'lenx', is_c, 4)}
    memcpy(A, Ai, NNelt * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &alpha, X, &inc, Y, &inc, A, &lda, 1',
                reset='memcpy(A, Ai, NNelt * sizeof(T));', touch=_out_touch('A'),
                free='    free(A); free(Ai); free(X); free(Y);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_trmv_trsv(name, is_c):
    dom = 'MKC((double)(N + 4), 0.0)' if is_c else 'MKR((double)(N + 4))'
    sig = ('const char *, const char *, const char *, const int *, '
           'const T *, const int *, T *, const int *, size_t, size_t, size_t')
    setup = f'''    int absi = inc < 0 ? -inc : inc, lda = N;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t NNelt = (size_t)N * (size_t)N;
    T *A = {_ALLOC}NNelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Xi = {_ALLOC}lenx * sizeof(T));
    {fill('A', 'NNelt', is_c, 2)}
    for (int d = 0; d < N; ++d) A[(size_t)d * N + d] = {dom};
    {fill('Xi', 'lenx', is_c, 3)}
    memcpy(X, Xi, lenx * sizeof(T));
{_kbuf_inc(['uplo', 'trans', 'diag'])}'''
    transes = ['N', 'T', 'C'] if is_c else ['N', 'T']
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
    const char transes[] = {_chars(transes)};
    const char diags[] = {_chars(['N', 'U'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
      for (size_t d = 0; d < sizeof(diags); ++d) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], transes[t], diags[d], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, char trans, char diag, int N, int inc', setup=setup,
                call='&uplo, &trans, &diag, &N, A, &lda, X, &inc, 1, 1, 1',
                reset='memcpy(X, Xi, lenx * sizeof(T));', touch=_out_touch('X'),
                free='    free(A); free(X); free(Xi);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# L2 banded
# ---------------------------------------------------------------------------
def build_gbmv(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const int *, const int *, const int *, const int *, '
           'const T *, const T *, const int *, const T *, const int *, '
           'const T *, T *, const int *, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int KL = 16, KU = 16, LDA = 33;
    int XL = (trans == 'N') ? N : M, YL = (trans == 'N') ? M : N;
    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(XL - 1) * (size_t)absi;
    const size_t leny = (size_t)1 + (size_t)(YL - 1) * (size_t)absi;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    T *A = {_ALLOC}Aelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}leny * sizeof(T));
    T *Yi = {_ALLOC}leny * sizeof(T));
    {fill('A', 'Aelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Yi', 'leny', is_c, 4)}
    memcpy(Y, Yi, leny * sizeof(T));
{_kbuf_inc(['trans'])}'''
    transes = ['N', 'T', 'C'] if is_c else ['N', 'T']
    sweep = f'''    const char transes[] = {_chars(transes)};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t t = 0; t < sizeof(transes); ++t) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(transes[t], sizes[i], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char trans, int M, int N, int inc', setup=setup,
                call='&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &inc, &beta, Y, &inc, 1',
                reset='memcpy(Y, Yi, leny * sizeof(T));', touch=_out_touch('Y'),
                free='    free(A); free(X); free(Y); free(Yi);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_sbmv_hbmv(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const int *, const int *, const T *, const T *, '
           'const int *, const T *, const int *, const T *, T *, const int *, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int K = 16, LDA = 17;
    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    T *A = {_ALLOC}Aelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}leny * sizeof(T));
    T *Yi = {_ALLOC}leny * sizeof(T));
    {fill('A', 'Aelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Yi', 'leny', is_c, 4)}
    memcpy(Y, Yi, leny * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &K, &alpha, A, &LDA, X, &inc, &beta, Y, &inc, 1',
                reset='memcpy(Y, Yi, leny * sizeof(T));', touch=_out_touch('Y'),
                free='    free(A); free(X); free(Y); free(Yi);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_tbmv_tbsv(name, is_c):
    dom = 'MKC((double)(K + 4), 0.0)' if is_c else 'MKR((double)(K + 4))'
    sig = ('const char *, const char *, const char *, const int *, const int *, '
           'const T *, const int *, T *, const int *, size_t, size_t, size_t')
    setup = f'''    int K = 16, LDA = 17;
    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    T *A = {_ALLOC}Aelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Xi = {_ALLOC}lenx * sizeof(T));
    {fill('A', 'Aelt', is_c, 2)}
    int diag_row = (uplo == 'U') ? K : 0;
    for (int j = 0; j < N; ++j) A[(size_t)j * LDA + diag_row] = {dom};
    {fill('Xi', 'lenx', is_c, 3)}
    memcpy(X, Xi, lenx * sizeof(T));
{_kbuf_inc(['uplo', 'trans', 'diag'])}'''
    transes = ['N', 'T', 'C'] if is_c else ['N', 'T']
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
    const char transes[] = {_chars(transes)};
    const char diags[] = {_chars(['N', 'U'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
      for (size_t d = 0; d < sizeof(diags); ++d) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], transes[t], diags[d], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, char trans, char diag, int N, int inc', setup=setup,
                call='&uplo, &trans, &diag, &N, &K, A, &LDA, X, &inc, 1, 1, 1',
                reset='memcpy(X, Xi, lenx * sizeof(T));', touch=_out_touch('X'),
                free='    free(A); free(X); free(Xi);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# L2 packed
# ---------------------------------------------------------------------------
def build_spr_hpr(name, is_c, is_h):
    Ta = 'RT' if is_h else ('CT' if is_c else 'RT')
    p7 = RL7 if is_h else (CL7 if is_c else RL7)
    sig = (f'const char *, const int *, const {Ta} *, const T *, const int *, '
           'T *, size_t')
    setup = f'''    {Ta} alpha = {p7};
    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t APelt = (size_t)N * (size_t)(N + 1) / 2;
    T *AP = {_ALLOC}APelt * sizeof(T));
    T *APi = {_ALLOC}APelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    {fill('APi', 'APelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    memcpy(AP, APi, APelt * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &alpha, X, &inc, AP, 1',
                reset='memcpy(AP, APi, APelt * sizeof(T));', touch=_out_touch('AP'),
                free='    free(AP); free(APi); free(X);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_spr2_hpr2(name, is_c):
    """Packed symmetric/Hermitian rank-2 update (spr2/hpr2). alpha is the
    element type (complex for hpr2). x and y share the stride `inc`."""
    Ta = 'CT' if is_c else 'RT'
    p7 = CL7 if is_c else RL7
    sig = (f'const char *, const int *, const {Ta} *, const T *, const int *, '
           'const T *, const int *, T *, size_t')
    setup = f'''    {Ta} alpha = {p7};
    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t APelt = (size_t)N * (size_t)(N + 1) / 2;
    T *AP = {_ALLOC}APelt * sizeof(T));
    T *APi = {_ALLOC}APelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}lenx * sizeof(T));
    {fill('APi', 'APelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Y', 'lenx', is_c, 4)}
    memcpy(AP, APi, APelt * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &alpha, X, &inc, Y, &inc, AP, 1',
                reset='memcpy(AP, APi, APelt * sizeof(T));', touch=_out_touch('AP'),
                free='    free(AP); free(APi); free(X); free(Y);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_spmv_hpmv(name, is_c):
    p7, p3 = (CL7, CL3) if is_c else (RL7, RL3)
    sig = ('const char *, const int *, const T *, const T *, const T *, '
           'const int *, const T *, T *, const int *, size_t')
    setup = f'''    T alpha = {p7}, beta = {p3};
    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t APelt = (size_t)N * (size_t)(N + 1) / 2;
    T *AP = {_ALLOC}APelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Y = {_ALLOC}leny * sizeof(T));
    T *Yi = {_ALLOC}leny * sizeof(T));
    {fill('AP', 'APelt', is_c, 2)}
    {fill('X', 'lenx', is_c, 3)}
    {fill('Yi', 'leny', is_c, 4)}
    memcpy(Y, Yi, leny * sizeof(T));
{_kbuf_inc(['uplo'])}'''
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, int N, int inc', setup=setup,
                call='&uplo, &N, &alpha, AP, X, &inc, &beta, Y, &inc, 1',
                reset='memcpy(Y, Yi, leny * sizeof(T));', touch=_out_touch('Y'),
                free='    free(AP); free(X); free(Y); free(Yi);', sweep=sweep)
    return is_c, 'void', sig, spec


def build_tpmv_tpsv(name, is_c):
    dom = 'MKC((double)(N + 4), 0.0)' if is_c else 'MKR((double)(N + 4))'
    sig = ('const char *, const char *, const char *, const int *, '
           'const T *, T *, const int *, size_t, size_t, size_t')
    setup = f'''    int absi = inc < 0 ? -inc : inc;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absi;
    const size_t APelt = (size_t)N * (size_t)(N + 1) / 2;
    T *AP = {_ALLOC}APelt * sizeof(T));
    T *X = {_ALLOC}lenx * sizeof(T));
    T *Xi = {_ALLOC}lenx * sizeof(T));
    {fill('AP', 'APelt', is_c, 2)}
    if (uplo == 'U') {{ size_t off = 0; for (int j = 0; j < N; ++j) {{ AP[off + j] = {dom}; off += (size_t)(j + 1); }} }}
    else {{ size_t off = 0; for (int j = 0; j < N; ++j) {{ AP[off] = {dom}; off += (size_t)(N - j); }} }}
    {fill('Xi', 'lenx', is_c, 3)}
    memcpy(X, Xi, lenx * sizeof(T));
{_kbuf_inc(['uplo', 'trans', 'diag'])}'''
    transes = ['N', 'T', 'C'] if is_c else ['N', 'T']
    sweep = f'''    const char uplos[] = {_chars(['U', 'L'])};
    const char transes[] = {_chars(transes)};
    const char diags[] = {_chars(['N', 'U'])};
{_strides_decl()}
{_size_iters(_l2_sizes(is_c), 'l2')}
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
      for (size_t d = 0; d < sizeof(diags); ++d) for (int k = 0; k < ninc; ++k)
        for (int i = 0; i < nsz; ++i)
            run_one(uplos[u], transes[t], diags[d], sizes[i], incs[k], itersarr[i], reps);'''
    spec = Spec(params='char uplo, char trans, char diag, int N, int inc', setup=setup,
                call='&uplo, &trans, &diag, &N, AP, X, &inc, 1, 1, 1',
                reset='memcpy(X, Xi, lenx * sizeof(T));', touch=_out_touch('X'),
                free='    free(AP); free(X); free(Xi);', sweep=sweep)
    return is_c, 'void', sig, spec


# ---------------------------------------------------------------------------
# L1 — unit stride only (matches the legacy L1 harnesses).
# ---------------------------------------------------------------------------
def _l1_sweep(is_c, call_args='sizes[i]'):
    return f'''{_size_iters(_l1_sizes(is_c), 'l1')}
    for (int i = 0; i < nsz; ++i)
        run_one({call_args}, itersarr[i], reps);'''


def build_axpy(name, is_c):
    p7 = CL7 if is_c else RL7
    sig = 'const int *, const T *, const T *, const int *, T *, const int *'
    setup = f'''    int one = 1;
    T alpha = {p7};
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    T *Y = {_ALLOC}(size_t)N * sizeof(T));
    T *Yi = {_ALLOC}(size_t)N * sizeof(T));
    {fill('X', 'N', is_c, 0)}
    {fill('Yi', 'N', is_c, 1)}
    memcpy(Y, Yi, (size_t)N * sizeof(T));
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup,
                call='&N, &alpha, X, &one, Y, &one',
                reset='memcpy(Y, Yi, (size_t)N * sizeof(T));', touch=_out_touch('Y'),
                free='    free(X); free(Y); free(Yi);', sweep=_l1_sweep(is_c))
    return is_c, 'void', sig, spec


def build_scal(name, is_c, alpha_real):
    Ta = 'RT' if alpha_real else ('CT' if is_c else 'RT')
    p7 = RL7 if alpha_real else (CL7 if is_c else RL7)
    sig = f'const int *, const {Ta} *, T *, const int *'
    setup = f'''    int one = 1;
    {Ta} alpha = {p7};
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    T *Xi = {_ALLOC}(size_t)N * sizeof(T));
    {fill('Xi', 'N', is_c, 0)}
    memcpy(X, Xi, (size_t)N * sizeof(T));
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, &alpha, X, &one',
                reset='memcpy(X, Xi, (size_t)N * sizeof(T));', touch=_out_touch('X'),
                free='    free(X); free(Xi);', sweep=_l1_sweep(is_c))
    return is_c, 'void', sig, spec


def build_copy_swap(name, is_c, swap):
    cst = '' if swap else 'const '
    sig = f'const int *, {cst}T *, const int *, T *, const int *'
    reset = ('memcpy(X, Xi, (size_t)N * sizeof(T)); memcpy(Y, Yi, (size_t)N * sizeof(T));'
             if swap else 'memcpy(Y, Yi, (size_t)N * sizeof(T));')
    setup = f'''    int one = 1;
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    T *Y = {_ALLOC}(size_t)N * sizeof(T));
    T *Xi = {_ALLOC}(size_t)N * sizeof(T));
    T *Yi = {_ALLOC}(size_t)N * sizeof(T));
    {fill('Xi', 'N', is_c, 0)}
    {fill('Yi', 'N', is_c, 1)}
    memcpy(X, Xi, (size_t)N * sizeof(T));
    memcpy(Y, Yi, (size_t)N * sizeof(T));
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, X, &one, Y, &one',
                reset=reset, touch=_out_touch('Y'),
                free='    free(X); free(Y); free(Xi); free(Yi);', sweep=_l1_sweep(is_c))
    return is_c, 'void', sig, spec


def build_dot(name, is_c, conj):
    ret = 'CT' if is_c else 'RT'
    kind = 'retcmplx' if is_c else 'retreal'
    sig = 'const int *, const T *, const int *, const T *, const int *'
    setup = f'''    int one = 1;
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    T *Y = {_ALLOC}(size_t)N * sizeof(T));
    {fill('X', 'N', is_c, 0)}
    {fill('Y', 'N', is_c, 1)}
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, X, &one, Y, &one',
                leg_kind=kind, free='    free(X); free(Y);', sweep=_l1_sweep(is_c))
    return is_c, ret, sig, spec


def build_reduce(name, is_c):
    """asum/nrm2 (and complex-in asum_c/nrm2_c): vector -> real scalar."""
    sig = 'const int *, const T *, const int *'
    setup = f'''    int one = 1;
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    {fill('X', 'N', is_c, 0)}
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, X, &one',
                leg_kind='retreal', free='    free(X);', sweep=_l1_sweep(is_c))
    return is_c, 'RT', sig, spec


def build_iamax(name, is_c):
    sig = 'const int *, const T *, const int *'
    setup = f'''    int one = 1;
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    {fill('X', 'N', is_c, 0)}
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, X, &one',
                leg_kind='retint', free='    free(X);', sweep=_l1_sweep(is_c))
    return is_c, 'int', sig, spec


def build_rot(name, is_c, real_cs):
    Tcs = 'RT' if real_cs else ('CT' if is_c else 'RT')
    p7 = RL7 if real_cs else (CL7 if is_c else RL7)
    p3 = RL3 if real_cs else (CL3 if is_c else RL3)
    sig = (f'const int *, T *, const int *, T *, const int *, '
           f'const {Tcs} *, const {Tcs} *')
    setup = f'''    int one = 1;
    {Tcs} c_ = {p7}, s_ = {p3};
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    T *Y = {_ALLOC}(size_t)N * sizeof(T));
    T *Xi = {_ALLOC}(size_t)N * sizeof(T));
    T *Yi = {_ALLOC}(size_t)N * sizeof(T));
    {fill('Xi', 'N', is_c, 0)}
    {fill('Yi', 'N', is_c, 1)}
    memcpy(X, Xi, (size_t)N * sizeof(T));
    memcpy(Y, Yi, (size_t)N * sizeof(T));
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, X, &one, Y, &one, &c_, &s_',
                reset='memcpy(X, Xi, (size_t)N * sizeof(T)); memcpy(Y, Yi, (size_t)N * sizeof(T));',
                touch=_out_touch('X'),
                free='    free(X); free(Y); free(Xi); free(Yi);', sweep=_l1_sweep(is_c))
    return is_c, 'void', sig, spec


def build_rotm(name):
    sig = 'const int *, T *, const int *, T *, const int *, const T *'
    setup = f'''    int one = 1;
    T PARAM[5] = {{ MKR(-1.0), {RL7}, {RL7}, {RL7}, {RL7} }};
    T *X = {_ALLOC}(size_t)N * sizeof(T));
    T *Y = {_ALLOC}(size_t)N * sizeof(T));
    T *Xi = {_ALLOC}(size_t)N * sizeof(T));
    T *Yi = {_ALLOC}(size_t)N * sizeof(T));
    {fill('Xi', 'N', False, 0)}
    {fill('Yi', 'N', False, 1)}
    memcpy(X, Xi, (size_t)N * sizeof(T));
    memcpy(Y, Yi, (size_t)N * sizeof(T));
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int N', setup=setup, call='&N, X, &one, Y, &one, PARAM',
                reset='memcpy(X, Xi, (size_t)N * sizeof(T)); memcpy(Y, Yi, (size_t)N * sizeof(T));',
                touch=_out_touch('X'),
                free='    free(X); free(Y); free(Xi); free(Yi);', sweep=_l1_sweep(False))
    return False, 'void', sig, spec


# ---------------------------------------------------------------------------
# scalar generators (fixed iteration count, per-call latency)
# ---------------------------------------------------------------------------
def build_cabs1(name):
    sig = 'const T *'
    setup = f'''    (void)nominal;
    T Z = {CL7};
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int nominal', setup=setup, call='&Z',
                leg_kind='retreal', size='0',
                sweep='    run_one(1, 100000, reps);')
    return True, 'RT', sig, spec


def build_rotg(name, is_c):
    Ta = 'CT' if is_c else 'RT'
    Tc = 'RT'
    Ts = Ta
    p7 = CL7 if is_c else RL7
    s0 = 'MKC(0.0, 0.0)' if is_c else 'MKR(0.0)'
    sig = f'{Ta} *, {Ta} *, {Tc} *, {Ts} *'
    setup = f'''    (void)nominal;
    {Ta} A0 = {p7}, B0 = {p7}, a, b;
    {Tc} c_ = MKR(0.0); {Ts} s_ = {s0};
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int nominal', setup=setup, call='&a, &b, &c_, &s_',
                reset='a = A0; b = B0;', size='0',
                sweep='    run_one(1, 100000, reps);')
    return is_c, 'void', sig, spec


def build_rotmg(name):
    sig = 'RT *, RT *, RT *, const RT *, RT *'
    setup = f'''    (void)nominal;
    RT D10 = {RL7}, D20 = {RL7}, X10 = {RL7}, Y1 = {RL7}, d1, d2, x1;
    RT PARAM[5];
    char kbuf[4]; kbuf[0] = '-'; kbuf[1] = 0;'''
    spec = Spec(params='int nominal', setup=setup, call='&d1, &d2, &x1, &Y1, PARAM',
                reset='d1 = D10; d2 = D20; x1 = X10;', size='0',
                sweep='    run_one(1, 100000, reps);')
    return False, 'void', sig, spec


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
_L3 = {'gemm', 'symm', 'hemm', 'syrk', 'herk', 'syr2k', 'her2k',
       'trmm', 'trsm', 'gemmtr'}
_L2 = {'gemv', 'ger', 'geru', 'gerc', 'symv', 'hemv', 'syr', 'her',
       'syr2', 'her2', 'spr2', 'hpr2',
       'trmv', 'trsv', 'gbmv', 'sbmv', 'hbmv', 'tbmv', 'tbsv',
       'spr', 'hpr', 'spmv', 'hpmv', 'tpmv', 'tpsv'}
_L1 = {'axpy', 'scal', 'cscal_r', 'copy', 'swap', 'dot', 'dotu', 'dotc',
       'asum', 'asum_c', 'nrm2', 'nrm2_c', 'iamax', 'rot', 'crot_r', 'rotm',
       'cabs1', 'rotg', 'rotmg'}
SUPPORTED = _L3 | _L2 | _L1


def build_spec(name):
    """(is_c, ret, sig, Spec) for a routine, or None if shape unsupported (yet)."""
    suffix, is_c = routine_shape(name)
    # L3
    if suffix == 'gemm':
        return build_gemm(name, is_c)
    if suffix in ('symm', 'hemm'):
        return build_symm_hemm(name, is_c or suffix == 'hemm')
    if suffix in ('syrk', 'herk'):
        return build_syrk_herk(name, is_c or suffix == 'herk', is_h=suffix == 'herk')
    if suffix in ('syr2k', 'her2k'):
        return build_syr2k_her2k(name, is_c or suffix == 'her2k', is_h=suffix == 'her2k')
    if suffix in ('trmm', 'trsm'):
        return build_trmm_trsm(name, is_c)
    if suffix == 'gemmtr':
        return build_gemmtr(name, is_c)
    # L2 dense
    if suffix == 'gemv':
        return build_gemv(name, is_c)
    if suffix in ('ger', 'geru', 'gerc'):
        return build_ger(name, is_c)
    if suffix in ('symv', 'hemv'):
        return build_symv_hemv(name, is_c or suffix == 'hemv')
    if suffix in ('syr', 'her'):
        return build_syr_her(name, is_c or suffix == 'her', is_h=suffix == 'her')
    if suffix in ('syr2', 'her2'):
        return build_syr2_her2(name, is_c or suffix == 'her2')
    if suffix in ('trmv', 'trsv'):
        return build_trmv_trsv(name, is_c)
    # L2 banded
    if suffix == 'gbmv':
        return build_gbmv(name, is_c)
    if suffix in ('sbmv', 'hbmv'):
        return build_sbmv_hbmv(name, is_c or suffix == 'hbmv')
    if suffix in ('tbmv', 'tbsv'):
        return build_tbmv_tbsv(name, is_c)
    # L2 packed
    if suffix in ('spr', 'hpr'):
        return build_spr_hpr(name, is_c or suffix == 'hpr', is_h=suffix == 'hpr')
    if suffix in ('spr2', 'hpr2'):
        return build_spr2_hpr2(name, is_c or suffix == 'hpr2')
    if suffix in ('spmv', 'hpmv'):
        return build_spmv_hpmv(name, is_c or suffix == 'hpmv')
    if suffix in ('tpmv', 'tpsv'):
        return build_tpmv_tpsv(name, is_c)
    # L1
    if suffix == 'axpy':
        return build_axpy(name, is_c)
    if suffix == 'scal':
        return build_scal(name, is_c, alpha_real=False)
    if suffix == 'cscal_r':
        return build_scal(name, True, alpha_real=True)
    if suffix == 'copy':
        return build_copy_swap(name, is_c, swap=False)
    if suffix == 'swap':
        return build_copy_swap(name, is_c, swap=True)
    if suffix == 'dot':
        return build_dot(name, is_c, conj=False)
    if suffix == 'dotu':
        return build_dot(name, True, conj=False)
    if suffix == 'dotc':
        return build_dot(name, True, conj=True)
    if suffix in ('asum', 'nrm2'):
        return build_reduce(name, is_c)
    if suffix in ('asum_c', 'nrm2_c'):
        return build_reduce(name, True)
    if suffix == 'iamax':
        return build_iamax(name, is_c)
    if suffix == 'rot':
        return build_rot(name, is_c, real_cs=False)
    if suffix == 'crot_r':
        return build_rot(name, True, real_cs=True)
    if suffix == 'rotm':
        return build_rotm(name)
    if suffix == 'cabs1':
        return build_cabs1(name)
    if suffix == 'rotg':
        return build_rotg(name, is_c)
    if suffix == 'rotmg':
        return build_rotmg(name)
    return None
