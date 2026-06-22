#!/usr/bin/env python3
"""check_int_boundary.py — enforce the "int is a boundary-only type" rule.

Inside the parallel BLAS overlay, numbers are ptrdiff_t, logical flags are bool,
and BLAS option letters (uplo/trans/diag/side) are char. The fixed-width Fortran
ABI types (`int` / `int64_t`) are allowed ONLY at the two boundaries, both of
which live in common/:

  * common/epblas_facade.h — the LP64 `name_` / ILP64 `name_64_` facades
                             (the public Fortran by-reference ABI).
  * common/blas_omp.h      — the OpenMP runtime wrapper (omp_get_max_threads()
                             returns int; widened to ptrdiff_t at the boundary).

Everything under kind10/ kind16/ multifloats/ must be free of bare `int`
declarations. This guard strips comments and string/char literals, then flags any
remaining bare `int` token. Run standalone or via ctest
(epblas_parallel_int_boundary_guard). Exit 0 = clean, 1 = violation.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "epblas-parallel"
DIRS = ["kind10", "kind16", "multifloats"]

# bare `int` type token, not glued into a longer identifier (int64_t, uint, …)
BARE_INT = re.compile(r"(?<![A-Za-z0-9_])int(?![A-Za-z0-9_])")


def strip_noise(text: str) -> str:
    """Blank out block comments, line comments, and string/char literals so the
    word `int` inside prose or text is not mistaken for a declaration. Replaces
    each with same-length blanks (newlines preserved) to keep line numbers."""
    out = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        c = text[i]
        if two == "/*":
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif two == "//":
            j = text.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 1
                j += 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def main() -> int:
    fail = False
    for d in DIRS:
        for f in sorted((SRC / d).rglob("*")):
            if f.suffix not in (".c", ".cpp", ".h"):
                continue
            code = strip_noise(f.read_text())
            for lineno, line in enumerate(code.splitlines(), 1):
                if BARE_INT.search(line):
                    if not fail:
                        pass
                    fail = True
                    rel = f.relative_to(ROOT)
                    print(f"VIOLATION: bare 'int' in {rel}:{lineno}")
                    print(f"    {line.strip()}")
    if fail:
        print()
        print("FAIL: 'int' is a boundary-only type. Use ptrdiff_t (sizes/indices/")
        print("      counts), bool (logical flags), or char (BLAS option letters).")
        print("      int/int64_t are allowed only in common/epblas_facade.h and")
        print("      common/blas_omp.h.")
        return 1
    print("OK: no bare 'int' in kind10/ kind16/ multifloats/ (boundary-only rule holds).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
