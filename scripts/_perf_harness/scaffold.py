"""Shared C-source scaffolding emitters.

Two pieces of boilerplate every harness shares:

  * BLAS_EXTERN <ret> <name>_(...);  / _migrated_(...);  → emit_externs
  * main() that reads BLAS_PERF_{ITERS,WARMUP}, parses sizes, prints the
    header, and drives a single-arg runner over each size → emit_main_linear

The L2 (incx/incy sweep) and L3 (multi-dim sweep) main() shapes are
emitter-specific — they stay inline in their emit_X function. Lifting
those would require parametrizing on the driver loop, which buys less
than it costs in clarity. The deeper C-side scaffolding (alloc/init/
timing/sink) lives inside the run_X body and is the target of the F5
follow-up PR (move into tests/epblas-parallel/perf/perf_common.h).
"""

L1_DEFAULT_SIZES = '64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536'


def sizes_body(real_sizes: tuple[int, ...], is_c: bool) -> str:
    """Comma-joined body for an L2/L3 `default_sizes[]` literal.

    A complex routine does ~4x the flops of its real twin at a given N (four
    real multiplies per complex multiply), so it runs ~4x longer per size and
    is the one that trips the per-binary wall-clock cap in the cmp5 sweep
    (ytrsm/ytrsv/ytpsv at the top size). Drop the largest size for complex so
    its wall-time stays comparable to the real routine; the real twin keeps the
    full range. Used as `{{{sizes_body(...)}}}` inside the emitters' f-strings.
    """
    sizes = real_sizes[:-1] if (is_c and len(real_sizes) > 1) else real_sizes
    return ', '.join(str(s) for s in sizes)


def emit_externs(name: str, ret_type: str, signature: str) -> str:
    """BLAS_EXTERN declaration pair for the subject + migrated symbols."""
    return (
        f'BLAS_EXTERN {ret_type} {name}_({signature});\n'
        f'BLAS_EXTERN {ret_type} {name}_migrated_({signature});\n'
    )


def emit_main_linear(runner: str, *,
                     sizes: str = L1_DEFAULT_SIZES,
                     iters: int = 200,
                     warmup: int = 20) -> str:
    """main() + default_sizes for an L1-style single-arg runner.

    Emits exactly the canonical body:

        static const int default_sizes[] = {sizes};
        int main(void) {
            int iters  = perf_env_int("BLAS_PERF_ITERS",  <iters>);
            int warmup = perf_env_int("BLAS_PERF_WARMUP", <warmup>);
            int sizes[32];
            int n = perf_parse_sizes(default_sizes,
                (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
            perf_print_header();
            for (int i = 0; i < n; ++i) <runner>(sizes[i], iters, warmup);
            return 0;
        }
    """
    return (
        f'static const int default_sizes[] = {{{sizes}}};\n'
        f'int main(void) {{\n'
        f'    int iters  = perf_env_int("BLAS_PERF_ITERS",  {iters});\n'
        f'    int warmup = perf_env_int("BLAS_PERF_WARMUP", {warmup});\n'
        f'    int sizes[32];\n'
        f'    int n = perf_parse_sizes(default_sizes,\n'
        f'        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);\n'
        f'    perf_print_header();\n'
        f'    for (int i = 0; i < n; ++i) {runner}(sizes[i], iters, warmup);\n'
        f'    return 0;\n'
        f'}}\n'
    )
