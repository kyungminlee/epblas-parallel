/* perf_common.h — shared helpers for kernel-isolated C perf harnesses.
 *
 * Why: each generated perf_<name>.{c,cpp} is independently linked with
 * -ffunction-sections -Wl,--gc-sections so the C overlay's symbol (subject)
 * and the migrated Fortran reference's symbol end up close in the binary
 * (avoids iTLB churn). The helpers below
 * stay header-only so each translation unit picks up only what it actually
 * uses; --gc-sections drops the rest.
 *
 * "Subject" here means whichever C overlay (epblas-openblas OR epblas-parallel)
 * is linked into the binary — perf_*.c sources are overlay-agnostic.
 *
 * Env knobs:
 *   BLAS_PERF_SIZES        comma list (default per-shape)
 *   BLAS_PERF_ITERS        timed iters per (shape-key, size)
 *   BLAS_PERF_WARMUP       warmup calls per (shape-key, size)
 *   BLAS_PERF_TIME_BUDGET  seconds; wall-time cap per timed loop (0 = unlimited).
 *                          When a timed loop reaches the budget it stops after
 *                          the current iteration rather than running the full
 *                          BLAS_PERF_ITERS — so a slow (routine,size) yields a
 *                          coarse-but-real number instead of overrunning a
 *                          harness timeout and being dropped wholesale. Subject
 *                          and migrated are timed in separate loops, so one
 *                          (key,size) config can take up to ~2x the budget plus
 *                          its (unbudgeted) warmup. The emitted `iters` column
 *                          reports the count actually run (the migrated loop's,
 *                          since it is timed last), flagging budget-limited rows.
 */
#ifndef PERF_COMMON_H
#define PERF_COMMON_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline double perf_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline int perf_env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || v <= 0) return dflt;
    return (int)v;
}

/* Non-negative seconds from an env var (0 / unset / malformed = dflt). Used
 * for the BLAS_PERF_TIME_BUDGET wall-time cap. */
static inline double perf_env_double(const char *name, double dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    char *end;
    double v = strtod(s, &end);
    if (end == s || v < 0.0) return dflt;
    return v;
}

/* Per-timed-loop wall-time budget in seconds; 0 means unlimited. */
static inline double perf_time_budget_s(void) {
    return perf_env_double("BLAS_PERF_TIME_BUDGET", 0.0);
}

static inline int perf_parse_sizes(const int *defaults, int n_defaults,
                                   int *out, int max)
{
    const char *s = getenv("BLAS_PERF_SIZES");
    if (!s || !*s) {
        int n = n_defaults < max ? n_defaults : max;
        for (int i = 0; i < n; ++i) out[i] = defaults[i];
        return n;
    }
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) out[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    }
    return n;
}

/* Comma-separated signed-int list from an env var. Used for
 * BLAS_PERF_INCX where negative strides are meaningful. */
static inline int perf_parse_int_list(const char *env_name,
                                      const int *defaults, int n_defaults,
                                      int *out, int max)
{
    const char *s = getenv(env_name);
    if (!s || !*s) {
        int n = n_defaults < max ? n_defaults : max;
        for (int i = 0; i < n; ++i) out[i] = defaults[i];
        return n;
    }
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    }
    return n;
}

/* Deterministic fill helpers — bounded magnitudes so x87/quadmath
 * doesn't drift into denormal/overflow ranges across many compounding
 * BLAS calls. Same seed = same sequence; we reset before each timed
 * block. */
static inline double perf_fill_double(size_t i, int salt) {
    /* small rational in [-1, 1]. NOTE: the centring subtraction MUST be signed
     * — `(v % 211u) - 105u` is unsigned, so every residue < 105 wraps to ~1.8e19
     * and divides to ~8.7e16, injecting ~1e16 garbage into half the fill. Benign
     * for single-pass kernels (matvec/axpy/dot: both legs see the same values),
     * but it makes triangular SOLVE inputs catastrophically ill-conditioned, so
     * the solution overflows over N steps and x87 inf-arithmetic inflates the
     * timing ~100x. Cast to signed before centring. */
    size_t v = (i * 2654435761u) ^ (size_t)(salt * 0x9e3779b9u);
    return (double)((long)(v % 211u) - 105) / 211.0;
}

static inline void perf_print_header(void) {
    /* Line-buffer stdout so every PERF_EMIT row is flushed on its newline.
     * When a runner executes a binary under `timeout` and a slow routine is
     * SIGTERM-killed at the cap, block-buffered output would be discarded and
     * the partial-output salvage would lose rows that were already computed
     * (e.g. ytrsm: all completed configs vanished on kill). Line-buffering
     * makes each emitted row durable before any kill. Called once per binary,
     * before the first stdout write, as setvbuf requires. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    /* Single line printed once per binary. Format chosen so a downstream
     * Python aggregator can split on whitespace. */
    /* "subject_ns" = bare wall time (ns/call) of the C-overlay routine under
     * test in this binary (epblas-openblas OR epblas-parallel, depending on
     * which archive was linked — perf_*.c sources are overlay-agnostic).
     * Reported as ns/call, NOT GF/s: smaller is faster. The trailing ratio is
     * subject/mig wall time (< 1.0 ⇒ the C overlay is faster than migrated). */
    printf("# routine            key      size    iters   subject_ns   migrated_ns   subject/mig\n");
}

/* t_subject / t_mg: wall-clock seconds per iter of the C-overlay routine
 * (subject) vs the migrated Fortran reference (mg). Caller doesn't care
 * which overlay — perf_*.c sources are overlay-agnostic. Emitted as bare
 * wall time in ns/call (smaller = faster). */
static inline void perf_emit(const char *routine, const char *key, int size,
                             int iters, double t_subject, double t_mg)
{
    double ns_subject = t_subject * 1e9;
    double ns_mg = t_mg * 1e9;
    double ratio = (t_mg > 0) ? t_subject / t_mg : 0;  /* < 1.0 ⇒ subject faster */
    printf("%-18s  %-7s  %6d  %6d  %12.1f  %13.1f  %8.3fx\n",
           routine, key, size, iters, ns_subject, ns_mg, ratio);
}

/* aligned_alloc that bumps the requested size up to a multiple of the
 * alignment (per the standard). */
static inline void *perf_aligned_alloc(size_t align, size_t bytes) {
    if (bytes == 0) return NULL;
    size_t rounded = (bytes + align - 1) & ~(align - 1);
    void *p = aligned_alloc(align, rounded);
    if (!p) {
        fprintf(stderr, "aligned_alloc failed (%zu bytes, align=%zu)\n",
                rounded, align);
        exit(2);
    }
    return p;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---- generated-harness scaffolding ----------------------------------
 *
 * Macros used by the perf_*.{c,cpp} harness files (originally generated;
 * the generator has been retired and the files are hand-maintained since).
 * Each generated harness ran 4–6 copies
 * of the same alloc / fill / reset / time / emit boilerplate; lifting
 * the patterns here keeps each run_* body to the parts that actually
 * differ (BLAS call signature, key formatting).
 *
 * The fill macros use token-paste on T so each target's <T>_FROM (whether
 * a #define or a static inline) is picked up automatically — keeps these
 * macros target-agnostic.
 */

#define PERF_ALLOC(T, n) \
    ((T *)perf_aligned_alloc(64, (size_t)(n) * sizeof(T)))

#define PERF_FILL_R(T, dst, n, seed) \
    do { for (size_t _i = 0; _i < (size_t)(n); ++_i) \
            (dst)[_i] = T##_FROM(perf_fill_double(_i, (seed))); \
    } while (0)

#define PERF_FILL_C(T, dst, n, seed) \
    do { for (size_t _i = 0; _i < (size_t)(n); ++_i) \
            (dst)[_i] = T##_FROM(perf_fill_double(_i, (seed)), \
                                 perf_fill_double(_i, (seed) + 131)); \
    } while (0)

#define PERF_RESET(dst, src, n, T) \
    memcpy((dst), (src), (size_t)(n) * sizeof(T))

/* Time a call. The call expression goes as the trailing variadic args so
 * its own commas don't get parsed as macro arg separators.
 *
 * BLAS_PERF_TIME_BUDGET caps the loop's wall time: when set, the loop stops
 * after the iteration that crosses the budget. `n_iters` must be an lvalue —
 * it is overwritten with the count actually run so the caller emits the true
 * averaging depth. When the budget is unset the original tight loop runs (no
 * per-iteration clock read, so the measurement is unperturbed). */
#define PERF_TIME(t_out, n_iters, /* call_stmt */ ...) \
    do { double _bud = perf_time_budget_s(); \
         double _t0 = perf_now_s(); int _done = 0; \
         if (_bud <= 0.0) { \
             for (int _it = 0; _it < (n_iters); ++_it) { __VA_ARGS__; } \
             _done = (n_iters); \
         } else { \
             for (int _it = 0; _it < (n_iters); ++_it) { \
                 __VA_ARGS__; ++_done; \
                 if (perf_now_s() - _t0 >= _bud) break; \
             } \
         } \
         double _t1 = perf_now_s(); \
         (t_out) = (_t1 - _t0) / (_done ? _done : 1); \
         (n_iters) = _done; \
    } while (0)

/* Per-call timing with reset between iters: keeps the reset out of the
 * timed window so a single-threaded memcpy can't Amdahl-mask multi-threaded
 * scaling. `reset_stmts` is a statement (possibly compound, e.g.
 * `PERF_RESET(...); PERF_RESET(...)`); the BLAS call goes as the variadic
 * tail so its commas don't trip the macro arg parser. */
/* As PERF_TIME, but resets between iters (kept out of the timed sum) and
 * honours BLAS_PERF_TIME_BUDGET. This variant already reads the clock twice
 * per iter, so the budget check (against wall time since the loop started,
 * resets included) costs nothing extra. `n_iters` must be an lvalue — it is
 * overwritten with the count actually run. */
#define PERF_TIME_PER_CALL(t_out, n_iters, reset_stmts, /* call_stmt */ ...) \
    do { double _bud = perf_time_budget_s(); \
         double _t_sum = 0, _t_start = perf_now_s(); int _done = 0; \
         for (int _it = 0; _it < (n_iters); ++_it) { \
             double _a = perf_now_s(); __VA_ARGS__; \
             double _b = perf_now_s(); \
             _t_sum += (_b - _a); ++_done; \
             if (_bud > 0.0 && _b - _t_start >= _bud) break; \
             reset_stmts; \
         } \
         (t_out) = _t_sum / (_done ? _done : 1); \
         (n_iters) = _done; \
    } while (0)

#define PERF_EMIT(routine, key, size, iters, t_subject, t_mg) \
    perf_emit((routine), (key), (size), (iters), (t_subject), (t_mg))

#endif /* PERF_COMMON_H */
