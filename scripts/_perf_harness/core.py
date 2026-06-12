"""Routine catalog and shape classifier shared by the dual-link harness.

Defines: KIND10/KIND16/MULTIFLOATS routine lists, CATALOG, routine_shape.
The per-target type info, preambles, and timing-loop boilerplate live with the
generator in scripts/_perf_harness/dual.py.
"""
from __future__ import annotations

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
    'qsyrk xsyrk xherk qsyr2k qsymm xsymm xhemm qgemv xgemv '
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
    'mnrm2 mwnrm2 '
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
