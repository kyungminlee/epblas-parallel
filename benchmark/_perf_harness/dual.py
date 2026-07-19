"""Dual-link (option-2) perf driver generator.

Emits ONE standalone C/C++ driver per routine that links the namespaced
par_/ob_/mig_ archives (built by benchmark/dual/nsbuild.sh) into a single process
and times all three legs per rep on the same buffers, ROTATING which leg runs
first each rep. This kills the cross-process layout/frequency bias the old
separate-binary cmp5 harness suffered, AND the in-process DVFS slot bias (a
fixed ob->par->mig order made par/mig always run in the prior leg's frequency/
thermal wake — a ~5-6% penalty on high-power omp4 cells that min-over-reps could
not cancel), while keeping the real linked routine (dispatch, thresholds, PLT)
and the threaded path (OMP_NUM_THREADS).

Output row format (one per (key, size) cell):

    <routine> <key> <N> <reset_ns> <par_ns> <ob_ns> <mig_ns> <par/ob> <par/mig> <ob/mig>

reset_ns is the per-iter cost of the RMW reset (memcpy) and is subtracted from
each leg; for read-mostly / value-return shapes there is no reset so it is ~0.
min-over-reps per leg; start-leg rotation so each leg's min is a cold-slot sample.

This module absorbed the routine signatures of the retired single-subject
generators (deleted along with the cross-process cmp5/gap5 harness); the
dual-link generator is now the only perf-driver generator in the tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core import routine_shape

# ---------------------------------------------------------------------------
# Per-family C config (typedefs RT/CT, element constructors, link libs).
# Type aliases are RT/CT (not R/C) so they never clash with buffer names C, A.
# Each driver picks `typedef <RT|CT> T;` for its element type.
# ---------------------------------------------------------------------------
@dataclass
class Family:
    key: str            # e | q | m
    ext: str            # c | cpp
    preamble: str
    link: str           # extra link libs


KIND10 = Family(
    key='e', ext='c', link='-lgfortran -lm',
    preamble='''#include <complex.h>
typedef long double RT;
typedef _Complex long double CT;
#define MKR(d)      ((RT)(double)(d))
#define MKC(re,im)  ((RT)(double)(re) + 1.0iL * (RT)(double)(im))
''')

KIND16 = Family(
    key='q', ext='c', link='-lgfortran -lquadmath -lm',
    preamble='''#include <quadmath.h>
typedef __float128 RT;
typedef _Complex float __attribute__((mode(TC))) CT;
#define MKR(d)      ((RT)(double)(d))
#define MKC(re,im)  ((CT)((RT)(double)(re) + 1.0i * (RT)(double)(im)))
''')

MULTIFLOATS = Family(
    key='m', ext='cpp', link='-lgfortran -lquadmath -lm',
    preamble='''typedef struct { double v[2]; } RT;
typedef struct { RT r; RT i; } CT;
static inline RT MKR(double d) { RT x; x.v[0] = d; x.v[1] = 0.0; return x; }
static inline CT MKC(double re, double im) { CT z; z.r = MKR(re); z.i = MKR(im); return z; }
''')

FAMILIES = {'kind10': KIND10, 'kind16': KIND16, 'multifloats': MULTIFLOATS}


def family_for(name: str) -> Family:
    """First-letter prefix → family. e/y → kind10, q/x → kind16, m/w → multifloats."""
    c = name[0] if name[0] != 'i' else name[1]
    if c in 'ey':
        return KIND10
    if c in 'qx':
        return KIND16
    if c in 'mw':
        return MULTIFLOATS
    raise ValueError(f"cannot map routine {name!r} to a family")


# ---------------------------------------------------------------------------
# iters per (size, work-class), so each cell does ~constant work.
# ---------------------------------------------------------------------------
_BUDGET = {'l3': 8_000_000, 'l2': 4_000_000, 'l1': 20_000_000}
_WORK = {'l3': lambda n: n ** 3, 'l2': lambda n: n ** 2, 'l1': lambda n: n}


def iters_for(size: int, klass: str) -> int:
    if klass == 'scalar':
        return 100_000
    return max(1, round(_BUDGET[klass] / _WORK[klass](size)))


# ---------------------------------------------------------------------------
# Fill helpers — identical buffers across all three legs is what matters.
# ---------------------------------------------------------------------------
def fill_r(buf: str, cnt: str, salt: int = 0) -> str:
    p = 101 + salt
    return (f'for (size_t i = 0; i < (size_t)({cnt}); ++i) '
            f'{buf}[i] = MKR(((long)(i % {p}) - {p // 2}) / 200.0);')


def fill_c(buf: str, cnt: str, salt: int = 0) -> str:
    p = 101 + salt
    q = 97 + salt
    return (f'for (size_t i = 0; i < (size_t)({cnt}); ++i) '
            f'{buf}[i] = MKC(((long)(i % {p}) - {p // 2}) / 200.0, '
            f'((long)(i % {q}) - {q // 2}) / 200.0);')


def fill(buf: str, cnt: str, is_c: bool, salt: int = 0) -> str:
    return fill_c(buf, cnt, salt) if is_c else fill_r(buf, cnt, salt)


# ---------------------------------------------------------------------------
# Spec for one routine shape. The timing loop is uniform; only these vary.
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    params: str          # run_one params, e.g. "char uplo, char trans, int N, int K"
    setup: str           # C body: allocs/fills/dims/key build (sets char kbuf[])
    call: str            # arglist string passed to every leg
    reset: str = ''      # per-iter reset stmt (RMW) or '' (read-mostly/value-return)
    leg_kind: str = 'void'   # 'void' | 'retreal' | 'retint'
    touch: str = '0UL'   # anti-DCE expr read after timing (void shapes read output)
    size: str = 'N'      # printed size value
    free: str = ''       # free stmts
    sweep: str = ''      # main() body that drives run_one over the grid


_HEADER = '''/* GENERATED by benchmark/gen_dual_harnesses.py — do not edit by hand.
 * Dual-link (par/ob/mig) single-process perf driver for {name}.
 *
 *   {cc} -O3 -DNDEBUG -ffp-contract=fast -march=native -fopenmp {src} \\
 *       lib_par_ns.a lib_ob_ns.a lib_mig_ns.a <refblas>.a {link} -o dual_{name}
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
{preamble}typedef {elem} T;

{externs}
static double now_s(void) {{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}}
static volatile unsigned long gsink = 0;
'''


def _extern(prefix: str, name: str, ret: str, sig: str) -> str:
    return f'extern {ret} {prefix}{name}_({sig});'


_RET_T = {'retint': 'int', 'retreal': 'RT', 'retcmplx': 'CT'}


def _legcall(prefix: str, name: str, spec: Spec) -> str:
    """The leg statement for one prefix (par_/ob_/mig_).

    Value-return shapes (dot/asum/nrm2/iamax/cabs1) capture the first 8 bytes
    of the result into the volatile gsink so the call can't be DCE'd; the leg
    cost is the call itself (no buffer reset).
    """
    fn = f'{prefix}{name}_'
    if spec.leg_kind == 'void':
        return f'{fn}({spec.call});'
    ret_t = _RET_T[spec.leg_kind]
    return (f'{{ {ret_t} _v = {fn}({spec.call}); '
            f'gsink ^= *(const unsigned long *)(const void *)&_v; }}')


def render(name: str, fam: Family, is_c: bool, ret: str, sig: str, spec: Spec) -> str:
    """Assemble a full dual-link driver source for one routine."""
    elem = 'CT' if is_c else 'RT'
    cc = 'g++' if fam.ext == 'cpp' else 'gcc'
    # The par/ob/mig BLAS entry points are unmangled `extern "C"` symbols in
    # every family (including the C++ multifloats archives); guard so the
    # prototypes keep C linkage when the driver itself is compiled as C++.
    protos = '\n'.join(_extern(p, name, ret, sig) for p in ('par_', 'ob_', 'mig_'))
    externs = ('#ifdef __cplusplus\nextern "C" {\n#endif\n'
               f'{protos}\n'
               '#ifdef __cplusplus\n}\n#endif')
    header = _HEADER.format(name=name, cc=cc, src=f'dual_{name}.{fam.ext}',
                            link=fam.link, preamble=fam.preamble, elem=elem,
                            externs=externs)

    reset = spec.reset
    leg_ob = _legcall('ob_', name, spec)
    leg_par = _legcall('par_', name, spec)
    leg_mig = _legcall('mig_', name, spec)

    run_one = f'''
static void run_one({spec.params}, int iters, int reps) {{
{spec.setup}
    double bp = 1e30, bo = 1e30, bg = 1e30, br = 1e30;
    /* ROTATE leg order across reps. The three legs run block-sequentially
     * within a rep, so a fixed ob->par->mig order would make par/mig always
     * run in ob's DVFS/thermal wake — a frequency penalty present in EVERY
     * rep that min-over-reps cannot cancel (~5-6% on high-power omp4 cells,
     * confirmed by order-swap A/B). Cycling the start leg each rep gives every
     * leg ~reps/3 turns in the cold slot-1 position; its min-over-reps then
     * comes from a slot-1 sample, so all three are compared cold-to-cold. */
    for (int r = 0; r < reps; ++r) {{
        double t0 = now_s();
        for (int it = 0; it < iters; ++it) {{ {reset} }}
        {{ double t = (now_s() - t0) / iters; if (t < br) br = t; }}
        for (int s = 0; s < 3; ++s) {{
            int leg = (r + s) % 3;
            t0 = now_s();
            if (leg == 0) {{
                for (int it = 0; it < iters; ++it) {{ {reset} {leg_ob} }}
                {{ double t = (now_s() - t0) / iters; if (t < bo) bo = t; }}
            }} else if (leg == 1) {{
                for (int it = 0; it < iters; ++it) {{ {reset} {leg_par} }}
                {{ double t = (now_s() - t0) / iters; if (t < bp) bp = t; }}
            }} else {{
                for (int it = 0; it < iters; ++it) {{ {reset} {leg_mig} }}
                {{ double t = (now_s() - t0) / iters; if (t < bg) bg = t; }}
            }}
        }}
    }}
    gsink ^= (unsigned long)({spec.touch});
    double np = bp - br, no = bo - br, ng = bg - br;
    printf("{name} %s %d %.1f %.1f %.1f %.1f %.4f %.4f %.4f\\n",
           kbuf, (int)({spec.size}), br * 1e9, np * 1e9, no * 1e9, ng * 1e9,
           (no > 0) ? np / no : 0.0, (ng > 0) ? np / ng : 0.0, (ng > 0) ? no / ng : 0.0);
{spec.free}
}}
'''

    main = f'''
int main(int argc, char **argv) {{
    setvbuf(stdout, NULL, _IOLBF, 0);
    int reps = (argc > 1) ? atoi(argv[1]) : 10;
    printf("# dual_{name} (par vs ob vs mig), reps=%d  OMP via env\\n", reps);
    printf("# routine key N reset_ns par_ns ob_ns mig_ns par/ob par/mig ob/mig\\n");
{spec.sweep}
    (void)gsink;
    return 0;
}}
'''
    return header + run_one + main
