#!/usr/bin/env python3
"""Generate kernel-isolated C perf harnesses for every parallel-BLAS overlay
routine.

One perf_<routine>.{c,cpp} per (target, routine) under
tests/epblas-parallel/perf/target_<target>/. CMake in
tests/epblas-parallel/CMakeLists.txt globs perf_*.{c,cpp} and links each
with -ffunction-sections / --gc-sections so overlay's and migrated's
symbols stay within a few KB of each other (no iTLB-spread artifact —
see findings doc Addendum 14).

Each emit_<shape> function takes (name, ti, is_complex) and returns
the full source for one harness. No shared abstraction other than:

    PROLOGUE  — common includes + type aliases
    MAIN_FMT  — main() wrapper that calls run_<symbol>(sizes...)

Per-shape: extern decls + run_<symbol>(...) function. Plain string
formatting; nothing clever.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PERF_DIR = REPO / "tests" / "epblas-parallel" / "perf"

# ---------------------------------------------------------------------------
# Per-target type info
# ---------------------------------------------------------------------------
@dataclass
class TypeInfo:
    target: str
    real_T: str
    cmplx_T: str
    file_ext: str       # 'c' or 'cpp'
    preamble: str       # extra includes / typedefs
    real_fill: str      # expression: "<T>_FILL(i, salt)"
    cmplx_fill: str
    real_lit_p7: str    # literal 0.7 in T
    real_lit_p3: str    # 0.3
    cmplx_lit_p7: str
    cmplx_lit_p3: str

# Fill expressions: take an index `i` and a salt `s` (both int-ish), produce
# a bounded scalar. perf_fill_double returns a small rational in [-1, 1].

# TODO(follow-up F5+F7): lift KIND10_PREAMBLE / KIND16_PREAMBLE / MULTIFLOATS_PREAMBLE
# typedef-and-BLAS_EXTERN boilerplate into tests/epblas-parallel/perf/perf_common.h
# (one header per target, included by every harness). At the same time, replace
# the `if ((double)(*((double*)&r)) == -123e30)` strict-aliasing sink pattern
# with an asm-memory clobber. Will shrink every emitted file by ~12 lines.

KIND10_PREAMBLE = '''
#include <complex.h>
#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef long double R10;
typedef _Complex long double C10;
#define R10_FROM(d) ((R10)(d))
#define C10_FROM(re, im) ((R10)(re) + 1.0iL * (R10)(im))
static inline R10 Tr_from_d(double d) { return (R10)d; }
static inline C10 Tc_from_d(double d) { return (C10)d; }
'''

KIND16_PREAMBLE = '''
#include <quadmath.h>
#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef __float128 Q16;
typedef _Complex float __attribute__((mode(TC))) X16;
#define Q16_FROM(d) ((Q16)(double)(d))
#define X16_FROM(re, im) ((X16)((Q16)(double)(re) + 1.0i * (Q16)(double)(im)))
static inline Q16 Tr_from_d(double d) { return (Q16)d; }
static inline X16 Tc_from_d(double d) { return (X16)((Q16)d); }
'''

MULTIFLOATS_PREAMBLE = '''
#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef struct { double v[2]; } MFR;     /* float64x2 layout (POD) */
typedef struct { MFR r; MFR i; } MFC;    /* complex64x2 layout (POD) */
static inline MFR MFR_FROM(double d) { MFR x; x.v[0] = d; x.v[1] = 0.0; return x; }
static inline MFC MFC_FROM(double re, double im) {
    MFC z;
    z.r.v[0] = re; z.r.v[1] = 0.0;
    z.i.v[0] = im; z.i.v[1] = 0.0;
    return z;
}
static inline MFR Tr_from_d(double d) { return MFR_FROM(d); }
static inline MFC Tc_from_d(double d) { return MFC_FROM(d, 0.0); }
'''

TYPES = {
    'kind10': TypeInfo(
        target='kind10',
        real_T='R10', cmplx_T='C10',
        file_ext='c',
        preamble=KIND10_PREAMBLE,
        real_fill='R10_FROM(perf_fill_double(i, s))',
        cmplx_fill='C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131))',
        real_lit_p7='R10_FROM(0.7)', real_lit_p3='R10_FROM(0.3)',
        cmplx_lit_p7='C10_FROM(0.7, 0.0)', cmplx_lit_p3='C10_FROM(0.3, 0.0)',
    ),
    'kind16': TypeInfo(
        target='kind16',
        real_T='Q16', cmplx_T='X16',
        file_ext='c',
        preamble=KIND16_PREAMBLE,
        real_fill='Q16_FROM(perf_fill_double(i, s))',
        cmplx_fill='X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131))',
        real_lit_p7='Q16_FROM(0.7)', real_lit_p3='Q16_FROM(0.3)',
        cmplx_lit_p7='X16_FROM(0.7, 0.0)', cmplx_lit_p3='X16_FROM(0.3, 0.0)',
    ),
    'multifloats': TypeInfo(
        target='multifloats',
        real_T='MFR', cmplx_T='MFC',
        file_ext='cpp',
        preamble=MULTIFLOATS_PREAMBLE,
        real_fill='MFR_FROM(perf_fill_double(i, s))',
        cmplx_fill='MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131))',
        real_lit_p7='MFR_FROM(0.7)', real_lit_p3='MFR_FROM(0.3)',
        cmplx_lit_p7='MFC_FROM(0.7, 0.0)', cmplx_lit_p3='MFC_FROM(0.3, 0.0)',
    ),
}

# ---------------------------------------------------------------------------
# Routine catalog (basenames per target, in src/CMakeLists.txt order).
# ---------------------------------------------------------------------------
KIND10 = (
    'egemm ygemm egemmtr ygemmtr etrsm ytrsm etrmm ytrmm '
    'esyrk ysyrk yherk esyr2k ysyr2k yher2k egemv ygemv eger ygeru ygerc '
    'esymv yhemv etrsv ytrsv esymm ysymm yhemm etrmv ytrmv '
    'esyr yher escal eaxpy ecopy eswap erot edot easum '
    'yscal yescal yaxpy ycopy yswap yerot ydotu ydotc '
    'eyasum egbmv ygbmv esbmv yhbmv espmv yhpmv etbmv ytbmv '
    'etbsv ytbsv etpmv ytpmv etpsv ytpsv espr yhpr '
    'ieamax iyamax erotg erotm erotmg yrotg '
    'ecabs1 enrm2 eynrm2'
).split()
KIND16 = (
    'qgemm xgemm qgemmtr xgemmtr qtrsm xtrsm qtrmm xtrmm '
    'qsyrk xsyrk xherk qsymm xsymm xhemm qgemv xgemv '
    'qger xgeru xgerc qsymv xhemv qtrsv xtrsv qtrmv xtrmv '
    'qsyr xher qscal qaxpy qcopy qswap qrot qdot qasum '
    'xscal xqscal xaxpy xcopy xswap xqrot xdotu xdotc '
    'qxasum qgbmv xgbmv qsbmv xhbmv qspmv xhpmv qtbmv xtbmv '
    'qtbsv xtbsv qtpmv xtpmv qtpsv xtpsv qspr xhpr '
    'iqamax ixamax qrotg qrotm qrotmg xrotg'
).split()
MULTIFLOATS = (
    'mgemm wgemm mgemmtr wgemmtr mtrsm wtrsm mtrmm wtrmm '
    'msyrk wsyrk wherk msymm wsymm whemm mgemv wgemv '
    'mger wgeru wgerc msymv whemv mtrsv wtrsv mtrmv wtrmv '
    'msyr wher mscal wscal wmscal maxpy waxpy mcopy wcopy '
    'mswap wswap mrot wmrot mdot masum mwasum wdotu wdotc '
    'mgbmv wgbmv msbmv whbmv mspmv whpmv mtbmv wtbmv '
    'mtbsv wtbsv mtpmv wtpmv mtpsv wtpsv mspr whpr '
    'imamax iwamax mrotg mrotm mrotmg wrotg'
).split()

CATALOG = {'kind10': KIND10, 'kind16': KIND16, 'multifloats': MULTIFLOATS}

def routine_shape(name: str) -> tuple[str, bool]:
    """(shape, is_complex) for a routine basename.

    Special cases:
      - i?amax: integer-return; shape='iamax'.
      - {eyasum, qxasum, mwasum}: complex-in real-out asum, shape='asum_c'.
      - {eynrm2, qxnrm2, mwnrm2}: complex-in real-out nrm2, shape='nrm2_c'.
      - {ecabs1, qcabs1, mcabs1}: scalar complex |Re|+|Im|, shape='cabs1'.
      - {yescal, xqscal, wmscal}: complex vec, real alpha, shape='cscal_r'.
      - {yerot, xqrot, wmrot}: complex vec, real (c, s), shape='crot_r'.
    """
    if name[0] == 'i':
        return 'iamax', name[1] in 'yxw'
    if name in ('eyasum', 'qxasum', 'mwasum'):
        return 'asum_c', True
    if name in ('eynrm2', 'qxnrm2', 'mwnrm2'):
        return 'nrm2_c', True
    if name in ('ecabs1', 'qcabs1', 'mcabs1'):
        return 'cabs1', True
    if name in ('yescal', 'xqscal', 'wmscal'):
        return 'cscal_r', True
    if name in ('yerot', 'xqrot', 'wmrot'):
        return 'crot_r', True
    suffix = name[1:]
    is_cmplx = name[0] in 'yxw'
    return suffix, is_cmplx

# ---------------------------------------------------------------------------
# Header emitted at the top of every harness.
# ---------------------------------------------------------------------------
GEN_SENTINEL = 'GENERATED-BY-gen_perf_harnesses'

PROLOGUE = '''/* ''' + GEN_SENTINEL + ''' — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for {ROUTINE} (overlay vs migrated).
 * Built per-executable with -ffunction-sections / --gc-sections.
 */
#include "../perf_common.h"
{TARGET_PREAMBLE}
'''

# ---------------------------------------------------------------------------
# Shape emitters. One per BLAS operation. Each returns a full C/C++ source.
# Each emitter inlines the symbol-extern declarations, the run_<symbol>(...)
# function, and a main() that drives default sizes.
# ---------------------------------------------------------------------------

def emit_unsupported(name: str, ti: TypeInfo, shape: str) -> str:
    return f'''/* AUTOGENERATED — shape '{shape}' not yet implemented for {name}. */
#include <stdio.h>
int main(void) {{
    fprintf(stdout, "# {name}: not implemented (shape={shape})\\n");
    return 0;
}}
'''

# -- L1 ----------------------------------------------------------------------

def emit_axpy(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '8.0 * (double)N' if is_c else '2.0 * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const int *, const {T} *, const {T} *, const int *, {T} *, const int *);
BLAS_EXTERN void {name}_migrated_(const int *, const {T} *, const {T} *, const int *, {T} *, const int *);

static void run_{name}(int N, int iters, int warmup) {{
    int one = 1;
    {T} alpha = {p7};
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Y = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int s = 0; s < 1; ++s) {{}}
    for (int i = 0; i < N; ++i) {{ int s = 0; X[i] = {fill}; }}
    for (int i = 0; i < N; ++i) {{ int s = 1; Yi[i] = {fill}; }}
    memcpy(Y, Yi, (size_t)N * sizeof({T}));

    for (int r = 0; r < warmup; ++r) {{
        {name}_(&N, &alpha, X, &one, Y, &one);
        memcpy(Y, Yi, (size_t)N * sizeof({T}));
        {name}_migrated_(&N, &alpha, X, &one, Y, &one);
        memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}

    memcpy(Y, Yi, (size_t)N * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_(&N, &alpha, X, &one, Y, &one);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);

    memcpy(Y, Yi, (size_t)N * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_migrated_(&N, &alpha, X, &one, Y, &one);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);

    double flops = {flops};
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y); free(Yi);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_{name}(sizes[i], iters, warmup);
    return 0;
}}
'''

def emit_copy_swap(name: str, ti: TypeInfo, is_c: bool, swap: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    fill = ti.cmplx_fill if is_c else ti.real_fill
    return f'''
BLAS_EXTERN void {name}_(const int *, {'const ' if not swap else ''}{T} *, const int *, {T} *, const int *);
BLAS_EXTERN void {name}_migrated_(const int *, {'const ' if not swap else ''}{T} *, const int *, {T} *, const int *);

static void run_{name}(int N, int iters, int warmup) {{
    int one = 1;
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Y = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; Xi[i] = {fill}; }}
    for (int i = 0; i < N; ++i) {{ int s = 1; Yi[i] = {fill}; }}
    memcpy(X, Xi, (size_t)N * sizeof({T}));
    memcpy(Y, Yi, (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&N, X, &one, Y, &one);
        memcpy(X, Xi, (size_t)N * sizeof({T}));
        memcpy(Y, Yi, (size_t)N * sizeof({T}));
        {name}_migrated_(&N, X, &one, Y, &one);
        memcpy(X, Xi, (size_t)N * sizeof({T}));
        memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep the reset memcpy OUT of the
     * timed window so a single-threaded reset doesn't Amdahl-cap the
     * measured MT scaling at large N. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&N, X, &one, Y, &one);
        double b = perf_now_s();
        t_sum += (b - a);
        {'memcpy(X, Xi, (size_t)N * sizeof(' + T + ')); ' if swap else ''}memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&N, X, &one, Y, &one);
        double b = perf_now_s();
        t_sum += (b - a);
        {'memcpy(X, Xi, (size_t)N * sizeof(' + T + ')); ' if swap else ''}memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    /* Bytes moved per call: copy=2N*sizeof(T), swap=4N*sizeof(T). Report
     * as "flops" for uniform formatting. */
    double flops = {'4.0' if swap else '2.0'} * (double)N * (double)sizeof({T});
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y); free(Xi); free(Yi);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_{name}(sizes[i], iters, warmup);
    return 0;
}}
'''

def emit_scal(name: str, ti: TypeInfo, is_c: bool, alpha_real: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    Ta = ti.real_T if alpha_real else T
    p7 = (ti.real_lit_p7 if alpha_real else
          (ti.cmplx_lit_p7 if is_c else ti.real_lit_p7))
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '6.0 * (double)N' if is_c else '1.0 * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const int *, const {Ta} *, {T} *, const int *);
BLAS_EXTERN void {name}_migrated_(const int *, const {Ta} *, {T} *, const int *);

static void run_{name}(int N, int iters, int warmup) {{
    int one = 1;
    {Ta} alpha = {p7};
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; Xi[i] = {fill}; }}
    memcpy(X, Xi, (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&N, &alpha, X, &one);
        memcpy(X, Xi, (size_t)N * sizeof({T}));
        {name}_migrated_(&N, &alpha, X, &one);
        memcpy(X, Xi, (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of the
     * timed window so it doesn't Amdahl-mask MT scaling. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&N, &alpha, X, &one);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&N, &alpha, X, &one);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Xi);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_{name}(sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L2 GEMV ------------------------------------------------------------------

def emit_gemv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Y  = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    for (size_t i = 0; i < (size_t)M*N; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i)       {{ int s = 3; X[i] = {fill}; }}
    for (size_t i = 0; i < leny; ++i)       {{ int s = 4; Yi[i] = {fill}; }}

    memcpy(Y, Yi, leny * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
        {name}_migrated_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
    }}

    memcpy(Y, Yi, leny * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);

    memcpy(Y, Yi, leny * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_migrated_(&trans, &M, &N, &alpha, A, &M, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);

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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *Ai = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Y  = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    for (size_t i = 0; i < (size_t)M*N; ++i) {{ int s = 2; Ai[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i)       {{ int s = 3; X[i] = {fill}; }}
    for (size_t i = 0; i < leny; ++i)       {{ int s = 4; Y[i] = {fill}; }}
    memcpy(A, Ai, (size_t)M * (size_t)N * sizeof({T}));

    for (int r = 0; r < warmup; ++r) {{
        {name}_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof({T}));
        {name}_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);

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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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

def emit_gemm(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '8.0 * (double)M * (double)N * (double)K' if is_c else '2.0 * (double)M * (double)N * (double)K'
    pairs = "['NN','TN','NT','TT','CN','NC']" if is_c else "['NN','TN','NT','TT']"
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const int *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const int *, const int *, const int *,
    const {T} *, const {T} *, const int *, const {T} *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t);

static void run_one(char ta, char tb, int M, int N, int K, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)K * sizeof({T}));
    {T} *B  = ({T} *)perf_aligned_alloc(64, (size_t)K * (size_t)N * sizeof({T}));
    {T} *C  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *Ci = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    int lda = M, ldb = K, ldc = M;
    for (size_t i = 0; i < (size_t)M*K; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)K*N; ++i) {{ int s = 3; B[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)M*N; ++i) {{ int s = 4; Ci[i] = {fill}; }}
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));

    for (int r = 0; r < warmup; ++r) {{
        {name}_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
        {name}_migrated_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    }}
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_migrated_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);

    double flops = {flops};
    char key[3] = {{ta, tb, 0}};
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{64, 128, 256, 512}};
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

def emit_dot(name: str, ti: TypeInfo, is_c: bool, conjugated: bool) -> str:
    """Real dot, complex dotu (conjugated=False), complex dotc (conjugated=True).

    All three return T by value at the C source level. The ABI handles
    the underlying call shape (register pair for kind10 _Complex long
    double; sret hidden pointer for >16-byte returns like _Complex
    __float128 and complex64x2). gcc and gfortran agree on the ABI for
    these types so a natural `T r = name_(...)` declaration matches.

    Anti-DCE: accumulate first-byte of `r` into a `volatile` sink across
    the timed loop. Without this gcc collapses the loop to one call.
    """
    T = ti.cmplx_T if is_c else ti.real_T
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '8.0 * (double)N' if is_c else '2.0 * (double)N'
    sig = f'const int *, const {T} *, const int *, const {T} *, const int *'
    # Anti-DCE sink: take a few bytes of `r` and XOR into a volatile counter.
    return f'''
BLAS_EXTERN {T} {name}_({sig});
BLAS_EXTERN {T} {name}_migrated_({sig});

static volatile unsigned long perf_sink = 0;

static inline void sink_T(const {T} *p) {{
    /* extract first 8 bytes of T into volatile sink */
    unsigned long w;
    memcpy(&w, (const void *)p, sizeof(w));
    perf_sink ^= w;
}}

static void run_one(int N, int iters, int warmup) {{
    int one = 1;
    {T} r;
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Y = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; X[i] = {fill}; }}
    for (int i = 0; i < N; ++i) {{ int s = 1; Y[i] = {fill}; }}
    for (int r2 = 0; r2 < warmup; ++r2) {{
        r = {name}_(&N, X, &one, Y, &one); sink_T(&r);
        r = {name}_migrated_(&N, X, &one, Y, &one); sink_T(&r);
    }}
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {{
        r = {name}_(&N, X, &one, Y, &one);
        sink_T(&r);
    }}
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);

    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {{
        r = {name}_migrated_(&N, X, &one, Y, &one);
        sink_T(&r);
    }}
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);

    double flops = {flops};
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}}
'''

def emit_l1_reduce(name: str, ti: TypeInfo, is_c: bool,
                   flops_per_elem_real: float) -> str:
    """L1 vector→real-scalar reduction. Shared by asum and nrm2.

    asum/nrm2 have the same scan shape: walk one vector, fold into a real
    accumulator. Only the per-element flop count differs (asum=1, nrm2=2).
    Complex variants double the per-element count (one fold per Re and Im).
    """
    T = ti.cmplx_T if is_c else ti.real_T
    R = ti.real_T
    fill = ti.cmplx_fill if is_c else ti.real_fill
    fpe = flops_per_elem_real * (2.0 if is_c else 1.0)
    flops = f'{fpe} * (double)N'
    sig = f'const int *, const {T} *, const int *'
    return f'''
BLAS_EXTERN {R} {name}_({sig});
BLAS_EXTERN {R} {name}_migrated_({sig});

static void run_one(int N, int iters, int warmup) {{
    int one = 1;
    {R} r;
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; X[i] = {fill}; }}
    for (int r2 = 0; r2 < warmup; ++r2) {{
        r = {name}_(&N, X, &one);
        r = {name}_migrated_(&N, X, &one);
    }}
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) r = {name}_(&N, X, &one);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) r = {name}_migrated_(&N, X, &one);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = {flops};
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    if ((double)(*((double*)&r)) == -123e30) {{ free(X); return; }}
    free(X);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}}
'''

def emit_iamax(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '2.0 * (double)N' if is_c else '1.0 * (double)N'
    return f'''
BLAS_EXTERN int {name}_(const int *, const {T} *, const int *);
BLAS_EXTERN int {name}_migrated_(const int *, const {T} *, const int *);

static void run_one(int N, int iters, int warmup) {{
    int one = 1, r = 0;
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; X[i] = {fill}; }}
    for (int r2 = 0; r2 < warmup; ++r2) {{
        r ^= {name}_(&N, X, &one);
        r ^= {name}_migrated_(&N, X, &one);
    }}
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) r ^= {name}_(&N, X, &one);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) r ^= {name}_migrated_(&N, X, &one);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = {flops};
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    if (r == -123) {{ free(X); return; }}
    free(X);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L2 symv/hemv (UPLO, N) --------------------------------------------------

def emit_symv_hemv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Y  = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    for (size_t i = 0; i < (size_t)N*N; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i)      {{ int s = 3; X[i] = {fill}; }}
    for (size_t i = 0; i < leny; ++i)      {{ int s = 4; Yi[i] = {fill}; }}
    memcpy(Y, Yi, leny * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
        {name}_migrated_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
    }}
    memcpy(Y, Yi, leny * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(Y, Yi, leny * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_migrated_(&uplo, &N, &alpha, A, &N, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *Ai = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    for (size_t i = 0; i < (size_t)N*N; ++i) {{ int s = 2; Ai[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i)      {{ int s = 3; X[i] = {fill}; }}
    memcpy(A, Ai, (size_t)N * (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof({T}));
        {name}_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    char key[16];
    if (incx == 1) {{
        key[0] = uplo; key[1] = 0;
    }} else {{
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }}
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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

def emit_spr_hpr(name: str, ti: TypeInfo, is_c: bool, is_h: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    Talpha = ti.real_T if is_h else T
    p7 = ti.real_lit_p7 if is_h else (ti.cmplx_lit_p7 if is_c else ti.real_lit_p7)
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '4.0 * (double)N * (double)N' if is_c else '1.0 * (double)N * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const {Talpha} *,
    const {T} *, const int *, {T} *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const {Talpha} *,
    const {T} *, const int *, {T} *, size_t);

static void run_one(char uplo, int N, int incx, int iters, int warmup) {{
    {Talpha} alpha = {p7};
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    {T} *AP  = ({T} *)perf_aligned_alloc(64, AP_LEN * sizeof({T}));
    {T} *APi = ({T} *)perf_aligned_alloc(64, AP_LEN * sizeof({T}));
    {T} *X   = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    for (size_t i = 0; i < AP_LEN; ++i) {{ int s = 2; APi[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i)   {{ int s = 3; X[i]   = {fill}; }}
    memcpy(AP, APi, AP_LEN * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &alpha, X, &incx, AP, 1);
        memcpy(AP, APi, AP_LEN * sizeof({T}));
        {name}_migrated_(&uplo, &N, &alpha, X, &incx, AP, 1);
        memcpy(AP, APi, AP_LEN * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&uplo, &N, &alpha, X, &incx, AP, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(AP, APi, AP_LEN * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&uplo, &N, &alpha, X, &incx, AP, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(AP, APi, AP_LEN * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    char key[16];
    if (incx == 1) {{
        key[0] = uplo; key[1] = 0;
    }} else {{
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }}
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(APi); free(X);
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

# -- L2 spmv/hpmv (UPLO, N) packed sym-mv ------------------------------------

def emit_spmv_hpmv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '8.0 * (double)N * (double)N' if is_c else '2.0 * (double)N * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const char *, const int *, const {T} *, const {T} *,
    const {T} *, const int *, const {T} *, {T} *, const int *, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const int *, const {T} *, const {T} *,
    const {T} *, const int *, const {T} *, {T} *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int incy, int iters, int warmup) {{
    {T} alpha = {p7}, beta = {p3};
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    {T} *AP = ({T} *)perf_aligned_alloc(64, AP_LEN * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Y  = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    for (size_t i = 0; i < AP_LEN; ++i) {{ int s = 2; AP[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i)   {{ int s = 3; X[i]  = {fill}; }}
    for (size_t i = 0; i < leny; ++i)   {{ int s = 4; Yi[i] = {fill}; }}
    memcpy(Y, Yi, leny * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
        {name}_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
    }}
    memcpy(Y, Yi, leny * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(Y, Yi, leny * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(X); free(Y); free(Yi);
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

# -- L2 trmv/trsv (UPLO, TRANS, DIAG, N) -------------------------------------

def emit_trmv_trsv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    /* Diagonally dominant for trsv stability */
    for (size_t i = 0; i < (size_t)N*N; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (int i = 0; i < N; ++i) {{
        size_t idx = (size_t)i * N + i;
        A[idx] = {from_d}((double)(N + 4));
    }}
    for (size_t i = 0; i < lenx; ++i) {{ int s = 3; Xi[i] = {fill}; }}
    memcpy(X, Xi, lenx * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof({T}));
        {name}_migrated_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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

def emit_tpmv_tpsv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '4.0 * (double)N * (double)N' if is_c else '1.0 * (double)N * (double)N'
    from_d = 'Tc_from_d' if is_c else 'Tr_from_d'
    return f'''
BLAS_EXTERN void {name}_(const char *, const char *, const char *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void {name}_migrated_(const char *, const char *, const char *, const int *,
    const {T} *, {T} *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int incx,
                    int iters, int warmup) {{
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    {T} *AP = ({T} *)perf_aligned_alloc(64, AP_LEN * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    for (size_t i = 0; i < AP_LEN; ++i) {{ int s = 2; AP[i] = {fill}; }}
    /* Force diagonal to ~N for stability of tpsv */
    if (uplo == 'U') {{
        size_t off = 0;
        for (int j = 0; j < N; ++j) {{ AP[off + j] = {from_d}((double)(N + 4)); off += (size_t)(j + 1); }}
    }} else {{
        size_t off = 0;
        for (int j = 0; j < N; ++j) {{ AP[off] = {from_d}((double)(N + 4)); off += (size_t)(N - j); }}
    }}
    for (size_t i = 0; i < lenx; ++i) {{ int s = 3; Xi[i] = {fill}; }}
    memcpy(X, Xi, lenx * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof({T}));
        {name}_migrated_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    char key[16];
    if (incx == 1) {{
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    }} else {{
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }}
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(X); free(Xi);
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
            for (int i = 0; i < n; ++i) run_one(uplo, trans, diag, sizes[i], incx, iters, warmup);
        }}
    }}
    return 0;
}}
'''

# -- L2 banded: gbmv (TRANS, M, N, KL, KU) ----------------------------------

def emit_gbmv(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)LDA * (size_t)N * sizeof({T}));
    const int XL = (trans == 'N') ? N : M;
    const int YL = (trans == 'N') ? M : N;
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(XL - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(YL - 1) * (size_t)absy;
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Y  = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    for (size_t i = 0; i < (size_t)LDA*N; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i) {{ int s = 3; X[i] = {fill}; }}
    for (size_t i = 0; i < leny; ++i) {{ int s = 4; Yi[i] = {fill}; }}
    memcpy(Y, Yi, leny * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
        {name}_migrated_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
    }}
    memcpy(Y, Yi, leny * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(Y, Yi, leny * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_migrated_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)LDA * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Y  = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, leny * sizeof({T}));
    for (size_t i = 0; i < (size_t)LDA*N; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < lenx; ++i) {{ int s = 3; X[i] = {fill}; }}
    for (size_t i = 0; i < leny; ++i) {{ int s = 4; Yi[i] = {fill}; }}
    memcpy(Y, Yi, leny * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
        {name}_migrated_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof({T}));
    }}
    memcpy(Y, Yi, leny * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(Y, Yi, leny * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {name}_migrated_(&uplo, &N, &K, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
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
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)LDA * (size_t)N * sizeof({T}));
    {T} *X  = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, lenx * sizeof({T}));
    for (size_t i = 0; i < (size_t)LDA*N; ++i) {{ int s = 2; A[i] = {fill}; }}
    /* diagonal at known row of band — large to stabilize tbsv */
    int diag_row = (uplo == 'U') ? K : 0;
    for (int j = 0; j < N; ++j) A[(size_t)j * LDA + diag_row] = {from_d}((double)(K + 4));
    for (size_t i = 0; i < lenx; ++i) {{ int s = 3; Xi[i] = {fill}; }}
    memcpy(X, Xi, lenx * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof({T}));
        {name}_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    char key[16];
    if (incx == 1) {{
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    }} else {{
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }}
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
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

def emit_symm_hemm(name: str, ti: TypeInfo, is_c: bool) -> str:
    T = ti.cmplx_T if is_c else ti.real_T
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    p3 = ti.cmplx_lit_p3 if is_c else ti.real_lit_p3
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)Asz * (size_t)Asz * sizeof({T}));
    {T} *B  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *C  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *Ci = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    int lda = Asz, ldb = M, ldc = M;
    for (size_t i = 0; i < (size_t)Asz*Asz; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)M*N; ++i)     {{ int s = 3; B[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)M*N; ++i)     {{ int s = 4; Ci[i] = {fill}; }}
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
        {name}_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    }}
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = {flops};
    char key[3] = {{side, uplo, 0}};
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{64, 128, 256, 512}};
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '4.0 * (double)N * (double)N * (double)K' if is_c else '1.0 * (double)N * (double)N * (double)K'
    transes = "['N', 'C']" if is_h else ("['N', 'T']")
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
    int lda = A_rows, ldc = N;
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)A_rows * (size_t)A_cols * sizeof({T}));
    {T} *C  = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *Ci = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    for (size_t i = 0; i < (size_t)A_rows*A_cols; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)N*N; ++i)           {{ int s = 4; Ci[i] = {fill}; }}
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
        {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    }}
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = {flops};
    char key[3] = {{uplo, trans, 0}};
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(C); free(Ci);
}}

static const int default_sizes[] = {{64, 128, 256, 512}};
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    int lda = A_rows, ldb = A_rows, ldc = N;
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)A_rows * (size_t)A_cols * sizeof({T}));
    {T} *B  = ({T} *)perf_aligned_alloc(64, (size_t)A_rows * (size_t)A_cols * sizeof({T}));
    {T} *C  = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *Ci = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    for (size_t i = 0; i < (size_t)A_rows*A_cols; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)A_rows*A_cols; ++i) {{ int s = 3; B[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)N*N; ++i)           {{ int s = 4; Ci[i] = {fill}; }}
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
        {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    }}
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = {flops};
    char key[3] = {{uplo, trans, 0}};
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{64, 128, 256, 512}};
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    int lda = Asz, ldb = M;
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)Asz * (size_t)Asz * sizeof({T}));
    {T} *B  = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    {T} *Bi = ({T} *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof({T}));
    for (size_t i = 0; i < (size_t)Asz*Asz; ++i) {{ int s = 2; A[i] = {fill}; }}
    /* diagonal dominance for trsm */
    for (int i = 0; i < Asz; ++i) A[(size_t)i * lda + i] = {from_d}((double)(Asz + 4));
    for (size_t i = 0; i < (size_t)M*N; ++i) {{ int s = 4; Bi[i] = {fill}; }}
    memcpy(B, Bi, (size_t)M * (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof({T}));
        {name}_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    char key[5] = {{side, uplo, trans, diag, 0}};
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(Bi);
}}

static const int default_sizes[] = {{64, 128, 256, 512}};
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
    fill = ti.cmplx_fill if is_c else ti.real_fill
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
    int lda = Arows, ldb = Brows, ldc = N;
    {T} *A  = ({T} *)perf_aligned_alloc(64, (size_t)Arows * (size_t)Acols * sizeof({T}));
    {T} *B  = ({T} *)perf_aligned_alloc(64, (size_t)Brows * (size_t)Bcols * sizeof({T}));
    {T} *C  = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    {T} *Ci = ({T} *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof({T}));
    for (size_t i = 0; i < (size_t)Arows*Acols; ++i) {{ int s = 2; A[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)Brows*Bcols; ++i) {{ int s = 3; B[i] = {fill}; }}
    for (size_t i = 0; i < (size_t)N*N; ++i)         {{ int s = 4; Ci[i] = {fill}; }}
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
        {name}_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    }}
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof({T}));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        {name}_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = {flops};
    char key[4] = {{uplo, ta, tb, 0}};
    perf_emit("{name}", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}}

static const int default_sizes[] = {{64, 128, 256, 512}};
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

def emit_rot(name: str, ti: TypeInfo, is_c: bool, real_cs: bool) -> str:
    """rot: x, y get rotated by (c, s). real_cs=True means c, s are real
    even when X, Y are complex (yerot/xqrot/wmrot)."""
    T = ti.cmplx_T if is_c else ti.real_T
    Tcs = ti.real_T if real_cs else T
    p7 = ti.real_lit_p7 if real_cs else (ti.cmplx_lit_p7 if is_c else ti.real_lit_p7)
    p3 = ti.real_lit_p3 if real_cs else (ti.cmplx_lit_p3 if is_c else ti.real_lit_p3)
    fill = ti.cmplx_fill if is_c else ti.real_fill
    flops = '12.0 * (double)N' if is_c else '6.0 * (double)N'
    return f'''
BLAS_EXTERN void {name}_(const int *, {T} *, const int *, {T} *, const int *,
    const {Tcs} *, const {Tcs} *);
BLAS_EXTERN void {name}_migrated_(const int *, {T} *, const int *, {T} *, const int *,
    const {Tcs} *, const {Tcs} *);

static void run_one(int N, int iters, int warmup) {{
    int one = 1;
    {Tcs} c_ = {p7}, s_ = {p3};
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Y = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; Xi[i] = {fill}; }}
    for (int i = 0; i < N; ++i) {{ int s = 1; Yi[i] = {fill}; }}
    memcpy(X, Xi, (size_t)N * sizeof({T}));
    memcpy(Y, Yi, (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&N, X, &one, Y, &one, &c_, &s_);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
        {name}_migrated_(&N, X, &one, Y, &one, &c_, &s_);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy resets out of the
     * timed window so they don't Amdahl-mask MT scaling. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&N, X, &one, Y, &one, &c_, &s_);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&N, X, &one, Y, &one, &c_, &s_);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = {flops};
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y); free(Xi); free(Yi);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L1 rotm: modified Givens apply ------------------------------------------

def emit_rotm(name: str, ti: TypeInfo) -> str:
    """rotm: real-only. PARAM(5)."""
    T = ti.real_T
    p7 = ti.real_lit_p7
    fill = ti.real_fill
    return f'''
BLAS_EXTERN void {name}_(const int *, {T} *, const int *, {T} *, const int *,
    const {T} *);
BLAS_EXTERN void {name}_migrated_(const int *, {T} *, const int *, {T} *, const int *,
    const {T} *);

static void run_one(int N, int iters, int warmup) {{
    int one = 1;
    {T} PARAM[5] = {{ {p7}, {p7}, {p7}, {p7}, {p7} }};
    PARAM[0] = Tr_from_d(-1.0); /* hflag=-1 → full matrix path */
    {T} *X = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Y = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Xi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    {T} *Yi = ({T} *)perf_aligned_alloc(64, (size_t)N * sizeof({T}));
    for (int i = 0; i < N; ++i) {{ int s = 0; Xi[i] = {fill}; }}
    for (int i = 0; i < N; ++i) {{ int s = 1; Yi[i] = {fill}; }}
    memcpy(X, Xi, (size_t)N * sizeof({T}));
    memcpy(Y, Yi, (size_t)N * sizeof({T}));
    for (int r = 0; r < warmup; ++r) {{
        {name}_(&N, X, &one, Y, &one, PARAM);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
        {name}_migrated_(&N, X, &one, Y, &one, PARAM);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    /* Per-call kernel-only timing — keep memcpy resets out of the
     * timed window so they don't Amdahl-mask MT scaling. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_(&N, X, &one, Y, &one, PARAM);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {{
        double a = perf_now_s();
        {name}_migrated_(&N, X, &one, Y, &one, PARAM);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof({T})); memcpy(Y, Yi, (size_t)N * sizeof({T}));
    }}
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 4.0 * (double)N;
    perf_emit("{name}", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y); free(Xi); free(Yi);
}}

static const int default_sizes[] = {{64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536}};
int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}}
'''

# -- L1 rotg / rotmg: scalar-only generators. Bench loop length is the
# effective N; we measure per-call latency.

def emit_cabs1(name: str, ti: TypeInfo) -> str:
    """cabs1: scalar |Re(z)| + |Im(z)| (complex in, real out). No N — looped
    single-call shape like rotg; flops is a per-call proxy."""
    C = ti.cmplx_T
    R = ti.real_T
    return f'''
BLAS_EXTERN {R} {name}_(const {C} *);
BLAS_EXTERN {R} {name}_migrated_(const {C} *);

static void run_one(int iters, int warmup) {{
    {C} Z = {ti.cmplx_lit_p7};
    {R} acc = Tr_from_d(0.0);
    for (int r = 0; r < warmup; ++r) {{
        acc += {name}_(&Z);
        acc += {name}_migrated_(&Z);
    }}
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) acc += {name}_(&Z);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) acc += {name}_migrated_(&Z);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    /* per-call proxy: 2 abs + 1 add */
    double flops = 3.0;
    perf_emit("{name}", "-", iters, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", iters, iters, flops, t_subject, t_mg);
    if ((double)(*((double*)&acc)) == -123e30) return;
}}

int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}}
'''

def emit_rotg(name: str, ti: TypeInfo, is_c: bool) -> str:
    """rotg: real takes 4 scalar args; complex takes 4 scalars (C is real)."""
    Ta = ti.cmplx_T if is_c else ti.real_T
    Tc = ti.real_T  # c is always real
    Ts = Ta         # s matches A's type
    p7 = ti.cmplx_lit_p7 if is_c else ti.real_lit_p7
    return f'''
BLAS_EXTERN void {name}_({Ta} *, {Ta} *, {Tc} *, {Ts} *);
BLAS_EXTERN void {name}_migrated_({Ta} *, {Ta} *, {Tc} *, {Ts} *);

static void run_one(int iters, int warmup) {{
    {Ta} A = {p7}, B = {p7};
    {Tc} C = Tr_from_d(0.0);
    {Ts} S = {('Tc_from_d(0.0)' if is_c else 'Tr_from_d(0.0)')};
    /* per call: regenerate fresh A, B inputs */
    for (int r = 0; r < warmup; ++r) {{
        {Ta} a = A, b = B; {name}_(&a, &b, &C, &S);
        a = A; b = B; {name}_migrated_(&a, &b, &C, &S);
    }}
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {{
        {Ta} a = A, b = B; {name}_(&a, &b, &C, &S);
    }}
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {{
        {Ta} a = A, b = B; {name}_migrated_(&a, &b, &C, &S);
    }}
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    /* report time per call as "flops" abuse: per-call flop count ~10. */
    double flops = 10.0;
    perf_emit("{name}", "-", iters, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", iters, iters, flops, t_subject, t_mg);
}}

int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}}
'''

def emit_rotmg(name: str, ti: TypeInfo) -> str:
    T = ti.real_T
    p7 = ti.real_lit_p7
    return f'''
BLAS_EXTERN void {name}_({T} *, {T} *, {T} *, const {T} *, {T} *);
BLAS_EXTERN void {name}_migrated_({T} *, {T} *, {T} *, const {T} *, {T} *);

static void run_one(int iters, int warmup) {{
    {T} D1 = {p7}, D2 = {p7}, X1 = {p7}, Y1 = {p7};
    {T} PARAM[5];
    for (int r = 0; r < warmup; ++r) {{
        {T} d1 = D1, d2 = D2, x1 = X1;
        {name}_(&d1, &d2, &x1, &Y1, PARAM);
        d1 = D1; d2 = D2; x1 = X1;
        {name}_migrated_(&d1, &d2, &x1, &Y1, PARAM);
    }}
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {{
        {T} d1 = D1, d2 = D2, x1 = X1;
        {name}_(&d1, &d2, &x1, &Y1, PARAM);
    }}
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {{
        {T} d1 = D1, d2 = D2, x1 = X1;
        {name}_migrated_(&d1, &d2, &x1, &Y1, PARAM);
    }}
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 20.0;
    perf_emit("{name}", "-", iters, iters, flops, t_subject, t_mg);
    perf_emit_json("{name}", "-", iters, iters, flops, t_subject, t_mg);
}}

int main(void) {{
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}}
'''

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def emit_routine(name: str, ti: TypeInfo) -> str:
    suffix, is_c = routine_shape(name)
    prologue = PROLOGUE.format(ROUTINE=name, TARGET_PREAMBLE=ti.preamble)
    if suffix == 'axpy':
        return prologue + emit_axpy(name, ti, is_c)
    if suffix == 'copy':
        return prologue + emit_copy_swap(name, ti, is_c, swap=False)
    if suffix == 'swap':
        return prologue + emit_copy_swap(name, ti, is_c, swap=True)
    if suffix == 'scal':
        return prologue + emit_scal(name, ti, is_c, alpha_real=False)
    if suffix == 'cscal_r':
        return prologue + emit_scal(name, ti, True, alpha_real=True)
    if suffix == 'gemv':
        return prologue + emit_gemv(name, ti, is_c)
    if suffix in ('ger', 'geru', 'gerc'):
        return prologue + emit_ger(name, ti, is_c)
    if suffix == 'gemm':
        return prologue + emit_gemm(name, ti, is_c)
    if suffix == 'dot':
        return prologue + emit_dot(name, ti, False, conjugated=False)
    if suffix == 'dotu':
        return prologue + emit_dot(name, ti, True, conjugated=False)
    if suffix == 'dotc':
        return prologue + emit_dot(name, ti, True, conjugated=True)
    if suffix == 'asum':
        return prologue + emit_l1_reduce(name, ti, is_c=False, flops_per_elem_real=1.0)
    if suffix == 'asum_c':
        return prologue + emit_l1_reduce(name, ti, is_c=True,  flops_per_elem_real=1.0)
    if suffix == 'nrm2':
        return prologue + emit_l1_reduce(name, ti, is_c=False, flops_per_elem_real=2.0)
    if suffix == 'nrm2_c':
        return prologue + emit_l1_reduce(name, ti, is_c=True,  flops_per_elem_real=2.0)
    if suffix == 'cabs1':
        return prologue + emit_cabs1(name, ti)
    if suffix == 'iamax':
        return prologue + emit_iamax(name, ti, is_c)
    if suffix == 'symv' or suffix == 'hemv':
        return prologue + emit_symv_hemv(name, ti, is_c)
    if suffix == 'syr':
        return prologue + emit_syr_her(name, ti, is_c=False, is_her=False)
    if suffix == 'her':
        return prologue + emit_syr_her(name, ti, is_c=True, is_her=True)
    if suffix == 'spr':
        return prologue + emit_spr_hpr(name, ti, is_c=False, is_h=False)
    if suffix == 'hpr':
        return prologue + emit_spr_hpr(name, ti, is_c=True, is_h=True)
    if suffix == 'spmv':
        return prologue + emit_spmv_hpmv(name, ti, is_c=False)
    if suffix == 'hpmv':
        return prologue + emit_spmv_hpmv(name, ti, is_c=True)
    if suffix == 'trmv' or suffix == 'trsv':
        return prologue + emit_trmv_trsv(name, ti, is_c)
    if suffix == 'tpmv' or suffix == 'tpsv':
        return prologue + emit_tpmv_tpsv(name, ti, is_c)
    if suffix == 'tbmv' or suffix == 'tbsv':
        return prologue + emit_tbmv_tbsv(name, ti, is_c)
    if suffix == 'gbmv':
        return prologue + emit_gbmv(name, ti, is_c)
    if suffix == 'sbmv':
        return prologue + emit_sbmv_hbmv(name, ti, is_c=False)
    if suffix == 'hbmv':
        return prologue + emit_sbmv_hbmv(name, ti, is_c=True)
    if suffix == 'symm':
        return prologue + emit_symm_hemm(name, ti, is_c=is_c)
    if suffix == 'hemm':
        return prologue + emit_symm_hemm(name, ti, is_c=True)
    if suffix == 'syrk':
        return prologue + emit_syrk_herk(name, ti, is_c=is_c, is_h=False)
    if suffix == 'herk':
        return prologue + emit_syrk_herk(name, ti, is_c=True, is_h=True)
    if suffix == 'syr2k':
        return prologue + emit_syr2k_her2k(name, ti, is_c=is_c, is_h=False)
    if suffix == 'her2k':
        return prologue + emit_syr2k_her2k(name, ti, is_c=True, is_h=True)
    if suffix == 'trmm' or suffix == 'trsm':
        return prologue + emit_trmm_trsm(name, ti, is_c)
    if suffix == 'gemmtr':
        return prologue + emit_gemmtr(name, ti, is_c)
    if suffix == 'rot':
        return prologue + emit_rot(name, ti, is_c=False, real_cs=False)
    if suffix == 'crot_r':
        return prologue + emit_rot(name, ti, is_c=True, real_cs=True)
    if suffix == 'rotm':
        return prologue + emit_rotm(name, ti)
    if suffix == 'rotg':
        return prologue + emit_rotg(name, ti, is_c)
    if suffix == 'rotmg':
        return prologue + emit_rotmg(name, ti)
    # rest: unsupported stub
    return prologue + emit_unsupported(name, ti, suffix)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='kind10,kind16,multifloats')
    ap.add_argument('--routines', default='')
    args = ap.parse_args()
    only = set(args.routines.split(',')) if args.routines else None

    written = 0
    skipped = []
    for tgt in args.targets.split(','):
        ti = TYPES[tgt]
        outdir = PERF_DIR / f'target_{tgt}'
        outdir.mkdir(parents=True, exist_ok=True)
        for name in CATALOG[tgt]:
            if only and name not in only:
                continue
            outpath = outdir / f'perf_{name}.{ti.file_ext}'
            # Skip files that don't contain our sentinel — they're
            # hand-written and the generator must not clobber them.
            if outpath.exists() and GEN_SENTINEL not in outpath.read_text():
                continue
            src = emit_routine(name, ti)
            outpath.write_text(src)
            written += 1
            if 'not implemented' in src:
                skipped.append((tgt, name))
    print(f'wrote {written} files; {len(skipped)} unsupported')
    if skipped:
        from collections import Counter
        # group by shape
        shapes = Counter()
        for tgt, name in skipped:
            s, _ = routine_shape(name)
            shapes[s] += 1
        for s, n in sorted(shapes.items(), key=lambda x: -x[1]):
            print(f'  {s}: {n}')

if __name__ == '__main__':
    main()
