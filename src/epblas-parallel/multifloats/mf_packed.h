/*
 * mf_packed.h — packed-triangular column base-index helpers, shared by the
 * packed matvec routines (mtpmv / wtpmv). &ap[kk_upper(j)] / &ap[kk_lower(j,N)]
 * is column j of the packed triangle stored contiguously, so a packed matvec
 * reuses the dense per-column kernel with these offsets. Integer index math
 * only — no DD types, so this is precision-agnostic and trivially perf-neutral.
 *
 * Upper: kk = j(j+1)/2,        diag at kk+j, off-diag rows 0..j-1 at kk+0..j-1.
 * Lower: kk = j*N - j(j-1)/2,  diag at kk+0,  rows j+1..N-1 at kk+1..N-1-j.
 */
#pragma once

#include <cstddef>

namespace mf_packed {

inline std::size_t kk_upper(std::ptrdiff_t j) {
    return static_cast<std::size_t>(j) * (j + 1) / 2;
}
inline std::size_t kk_lower(std::ptrdiff_t j, std::ptrdiff_t n) {
    return static_cast<std::size_t>(j) * n - static_cast<std::size_t>(j) * (j - 1) / 2;
}

}  // namespace mf_packed
