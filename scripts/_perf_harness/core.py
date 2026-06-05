"""Core types, target preambles, routine catalog, and per-target PROLOGUE.

Defines: TypeInfo, TYPES, KIND10/KIND16/MULTIFLOATS routine lists, CATALOG,
routine_shape, GEN_SENTINEL, PROLOGUE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
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
    'iqamax ixamax qrotg qrotm qrotmg xrotg '
    'qnrm2 qxnrm2'
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
