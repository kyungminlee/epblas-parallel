/* blas_char.h — shared BLAS flag-character helper for the parallel overlay.
 *
 * BLAS option arguments (uplo/trans/diag/side) are single flag chars that the
 * standard allows in either case ('u' == 'U'). Every routine normalises them
 * with the same toupper idiom before branching. That one-liner used to be
 * copy-pasted as a file-local `static inline char up(...)` in ~60 TUs (in two
 * spellings — by value and by pointer) and open-coded as
 * `(char)toupper((unsigned char)x)` in ~90 more. This is the single definition.
 *
 * The argument and result are flag CHARS, not ints: decode is the one place a
 * BLAS routine handles option letters, and it stays in the char domain. Callers
 * that hold the flag by pointer pass the deref (`blas_up(*p)`); the result is
 * compared against literal chars ('U'/'L'/'N'/'T'/'C'/'R') or reduced to a bool.
 */
#ifndef EPBLAS_PARALLEL_BLAS_CHAR_H
#define EPBLAS_PARALLEL_BLAS_CHAR_H

#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline char blas_up(char c) { return (char)toupper((unsigned char)c); }

#ifdef __cplusplus
}
#endif

#endif /* EPBLAS_PARALLEL_BLAS_CHAR_H */
