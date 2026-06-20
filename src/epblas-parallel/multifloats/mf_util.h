/*
 * mf_util.h — tiny shared scalar helpers for the multifloats BLAS sources.
 *
 * up(): uppercase a BLAS character flag (uplo/trans/diag/side). 56 .cpp files
 * each re-derived the identical `(char)toupper((unsigned char)*p)` one-liner
 * (52 in static_cast spelling, 4 in C-cast spelling); they all collapse onto
 * this one inline. Cold (one call per BLAS invocation, never in a SIMD loop),
 * so it's perf-neutral — header-inline, same codegen as the local copies.
 *
 * Distinct from mf_pred.h (DD limb predicates) and the SIMD-DD arithmetic in
 * mf_simd_fast.h / mf_simd_exact.h.
 */
#pragma once

#include <cctype>

namespace mf_util {

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

}  // namespace mf_util
