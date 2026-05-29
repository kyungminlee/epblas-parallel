"""Shape → emitter dispatch table and the emit_routine entry point."""
from .core import PROLOGUE, TypeInfo, routine_shape
from .emit_l1 import (
    emit_unsupported, emit_axpy, emit_copy_swap, emit_scal, emit_dot,
    emit_l1_reduce, emit_iamax, emit_rot, emit_rotm, emit_cabs1,
    emit_rotg, emit_rotmg,
)
from .emit_l2_dense  import emit_gemv, emit_ger, emit_symv_hemv, emit_syr_her, emit_trmv_trsv
from .emit_l2_banded import emit_gbmv, emit_sbmv_hbmv, emit_tbmv_tbsv
from .emit_l2_packed import emit_spr_hpr, emit_spmv_hpmv, emit_tpmv_tpsv
from .emit_l3 import (
    emit_gemm, emit_symm_hemm, emit_syrk_herk, emit_syr2k_her2k,
    emit_trmm_trsm, emit_gemmtr,
)

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
# shape suffix → (emitter, **kwargs that the emitter takes beyond name+ti+is_c).
# For shapes whose emitter is shared across two BLAS names (e.g. symv/hemv,
# trmv/trsv), both shapes point at the same emitter. Hard-coded flags like
# is_c=True for h-routines preserve the historical behaviour of overriding
# whatever routine_shape returned.
EMITTERS = {
    'axpy':    lambda n, t, c: emit_axpy(n, t, c),
    'copy':    lambda n, t, c: emit_copy_swap(n, t, c, swap=False),
    'swap':    lambda n, t, c: emit_copy_swap(n, t, c, swap=True),
    'scal':    lambda n, t, c: emit_scal(n, t, c, alpha_real=False),
    'cscal_r': lambda n, t, c: emit_scal(n, t, True, alpha_real=True),
    'gemv':    lambda n, t, c: emit_gemv(n, t, c),
    'ger':     lambda n, t, c: emit_ger(n, t, c),
    'geru':    lambda n, t, c: emit_ger(n, t, c),
    'gerc':    lambda n, t, c: emit_ger(n, t, c),
    'gemm':    lambda n, t, c: emit_gemm(n, t, c),
    'dot':     lambda n, t, c: emit_dot(n, t, False, conjugated=False),
    'dotu':    lambda n, t, c: emit_dot(n, t, True,  conjugated=False),
    'dotc':    lambda n, t, c: emit_dot(n, t, True,  conjugated=True),
    'asum':    lambda n, t, c: emit_l1_reduce(n, t, is_c=False, flops_per_elem_real=1.0),
    'asum_c':  lambda n, t, c: emit_l1_reduce(n, t, is_c=True,  flops_per_elem_real=1.0),
    'nrm2':    lambda n, t, c: emit_l1_reduce(n, t, is_c=False, flops_per_elem_real=2.0),
    'nrm2_c':  lambda n, t, c: emit_l1_reduce(n, t, is_c=True,  flops_per_elem_real=2.0),
    'cabs1':   lambda n, t, c: emit_cabs1(n, t),
    'iamax':   lambda n, t, c: emit_iamax(n, t, c),
    'symv':    lambda n, t, c: emit_symv_hemv(n, t, c),
    'hemv':    lambda n, t, c: emit_symv_hemv(n, t, c),
    'syr':     lambda n, t, c: emit_syr_her(n, t, is_c=False, is_her=False),
    'her':     lambda n, t, c: emit_syr_her(n, t, is_c=True,  is_her=True),
    'spr':     lambda n, t, c: emit_spr_hpr(n, t, is_c=False, is_h=False),
    'hpr':     lambda n, t, c: emit_spr_hpr(n, t, is_c=True,  is_h=True),
    'spmv':    lambda n, t, c: emit_spmv_hpmv(n, t, is_c=False),
    'hpmv':    lambda n, t, c: emit_spmv_hpmv(n, t, is_c=True),
    'trmv':    lambda n, t, c: emit_trmv_trsv(n, t, c),
    'trsv':    lambda n, t, c: emit_trmv_trsv(n, t, c),
    'tpmv':    lambda n, t, c: emit_tpmv_tpsv(n, t, c),
    'tpsv':    lambda n, t, c: emit_tpmv_tpsv(n, t, c),
    'tbmv':    lambda n, t, c: emit_tbmv_tbsv(n, t, c),
    'tbsv':    lambda n, t, c: emit_tbmv_tbsv(n, t, c),
    'gbmv':    lambda n, t, c: emit_gbmv(n, t, c),
    'sbmv':    lambda n, t, c: emit_sbmv_hbmv(n, t, is_c=False),
    'hbmv':    lambda n, t, c: emit_sbmv_hbmv(n, t, is_c=True),
    'symm':    lambda n, t, c: emit_symm_hemm(n, t, is_c=c),
    'hemm':    lambda n, t, c: emit_symm_hemm(n, t, is_c=True),
    'syrk':    lambda n, t, c: emit_syrk_herk(n, t, is_c=c,    is_h=False),
    'herk':    lambda n, t, c: emit_syrk_herk(n, t, is_c=True, is_h=True),
    'syr2k':   lambda n, t, c: emit_syr2k_her2k(n, t, is_c=c,    is_h=False),
    'her2k':   lambda n, t, c: emit_syr2k_her2k(n, t, is_c=True, is_h=True),
    'trmm':    lambda n, t, c: emit_trmm_trsm(n, t, c),
    'trsm':    lambda n, t, c: emit_trmm_trsm(n, t, c),
    'gemmtr':  lambda n, t, c: emit_gemmtr(n, t, c),
    'rot':     lambda n, t, c: emit_rot(n, t, is_c=False, real_cs=False),
    'crot_r':  lambda n, t, c: emit_rot(n, t, is_c=True,  real_cs=True),
    'rotm':    lambda n, t, c: emit_rotm(n, t),
    'rotg':    lambda n, t, c: emit_rotg(n, t, c),
    'rotmg':   lambda n, t, c: emit_rotmg(n, t),
}

def emit_routine(name: str, ti: TypeInfo) -> str:
    suffix, is_c = routine_shape(name)
    prologue = PROLOGUE.format(ROUTINE=name, TARGET_PREAMBLE=ti.preamble)
    emitter = EMITTERS.get(suffix)
    if emitter is None:
        return prologue + emit_unsupported(name, ti, suffix)
    return prologue + emitter(name, ti, is_c)
