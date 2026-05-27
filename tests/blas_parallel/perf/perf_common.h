/* perf_common.h — shared helpers for kernel-isolated C perf harnesses.
 *
 * Why: each generated perf_<name>.{c,cpp} is independently linked with
 * -ffunction-sections -Wl,--gc-sections so the C overlay's symbol (subject)
 * and the migrated Fortran reference's symbol end up close in the binary
 * (avoids iTLB churn — see findings doc Addendum 14). The helpers below
 * stay header-only so each translation unit picks up only what it actually
 * uses; --gc-sections drops the rest.
 *
 * "Subject" here means whichever C overlay (epopenblas OR parallel-blas)
 * is linked into the binary — perf_*.c sources are overlay-agnostic.
 *
 * Env knobs:
 *   BLAS_PERF_SIZES     comma list (default per-shape)
 *   BLAS_PERF_ITERS     timed iters per (shape-key, size)
 *   BLAS_PERF_WARMUP    warmup calls per (shape-key, size)
 *   BLAS_PERF_JSON      write JSON results to this path (append mode)
 *   BLAS_PERF_LABEL     extra label printed alongside routine name
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
    /* small rational in [-1, 1]. */
    size_t v = (i * 2654435761u) ^ (size_t)(salt * 0x9e3779b9u);
    return (double)((v % 211u) - 105u) / 211.0;
}

static inline void perf_print_header(void) {
    /* Single line printed once per binary. Format chosen so a downstream
     * Python aggregator can split on whitespace. */
    /* "subject_GFs" = GF/s of the C-overlay routine under test in this
     * binary (epopenblas OR parallel-blas, depending on which archive was
     * linked — perf_*.c sources are overlay-agnostic). */
    printf("# routine            key      size    iters   subject_GFs   migrated_GFs   mig/subject\n");
}

/* t_subject / t_mg: wall-clock seconds per iter of the C-overlay routine
 * (subject) vs the migrated Fortran reference (mg). Caller doesn't care
 * which overlay — perf_*.c sources are overlay-agnostic. */
static inline void perf_emit(const char *routine, const char *key, int size,
                             int iters, double flops, double t_subject, double t_mg)
{
    double g_subject = (t_subject > 0) ? flops / t_subject / 1e9 : 0;
    double g_mg = (t_mg > 0) ? flops / t_mg / 1e9 : 0;
    double ratio = (t_subject > 0) ? t_mg / t_subject : 0;
    printf("%-18s  %-7s  %6d  %6d  %12.4f  %13.4f  %8.3fx\n",
           routine, key, size, iters, g_subject, g_mg, ratio);
}

/* JSON line emitter, optional. One JSON object per (routine, key, size).
 * Caller writes to BLAS_PERF_JSON in append mode if set.
 *
 * JSON keys:
 *   t_subject, gflops_subject  — C overlay (epopenblas OR parallel-blas; varies
 *                                by which archive is linked into the binary)
 *   t_mg, gflops_mg            — migrated Fortran reference
 *   ratio                      — subject GF/s ÷ migrated GF/s
 */
static inline void perf_emit_json(const char *routine, const char *key,
                                  int size, int iters, double flops,
                                  double t_subject, double t_mg)
{
    const char *path = getenv("BLAS_PERF_JSON");
    if (!path || !*path) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    double g_subject = (t_subject > 0) ? flops / t_subject / 1e9 : 0;
    double g_mg = (t_mg > 0) ? flops / t_mg / 1e9 : 0;
    double ratio = (t_subject > 0) ? t_mg / t_subject : 0;
    fprintf(f, "{\"routine\":\"%s\",\"key\":\"%s\",\"size\":%d,\"iters\":%d,"
               "\"t_subject\":%.6e,\"t_mg\":%.6e,\"gflops_subject\":%.4f,"
               "\"gflops_mg\":%.4f,\"ratio\":%.4f}\n",
            routine, key, size, iters, t_subject, t_mg, g_subject, g_mg, ratio);
    fclose(f);
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

#endif /* PERF_COMMON_H */
