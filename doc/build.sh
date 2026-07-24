#!/usr/bin/env bash
# Build the epblas-parallel documentation: Doxygen (XML) -> Sphinx/Breathe (HTML).
#
# One-time setup:
#     uv venv doc/.venv
#     uv pip install --python doc/.venv -r doc/requirements.txt
#     # plus the `doxygen` system package on PATH
#
# Usage:  ./doc/build.sh   (run from the repo root or from doc/)
set -euo pipefail

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DOC_DIR/.." && pwd)"
VENV="$DOC_DIR/.venv"
SPHINX="$VENV/bin/sphinx-build"

if ! command -v doxygen >/dev/null 2>&1; then
    echo "error: doxygen not found on PATH (install the system package)." >&2
    exit 1
fi
if [[ ! -x "$SPHINX" ]]; then
    echo "error: $SPHINX not found. Run:" >&2
    echo "    uv venv doc/.venv && uv pip install --python doc/.venv -r doc/requirements.txt" >&2
    exit 1
fi

echo ">> configure (VERSION -> conf.py, Doxyfile, version.h)"
EP_VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
MAJOR="${EP_VERSION%%.*}"; REST="${EP_VERSION#*.}"
MINOR="${REST%%.*}"; PATCH="${REST#*.}"
sed "s/@PROJECT_VERSION@/$EP_VERSION/g" "$DOC_DIR/conf.py.in"  > "$DOC_DIR/conf.py"
sed "s/@PROJECT_VERSION@/$EP_VERSION/g" "$DOC_DIR/Doxyfile.in" > "$DOC_DIR/Doxyfile"
# The public API surface is the generated version.h; materialize it from its
# template so Doxygen has something real to parse (the VERSION file is the
# single version source, no CMake configure needed).
mkdir -p "$DOC_DIR/_doxygen/include/epblas-parallel"
sed -e "s|@PROJECT_VERSION_MAJOR@|${MAJOR}|" \
    -e "s|@PROJECT_VERSION_MINOR@|${MINOR}|" \
    -e "s|@PROJECT_VERSION_PATCH@|${PATCH}|" \
    -e "s|@PROJECT_VERSION@|${EP_VERSION}|" \
    "$ROOT_DIR/include/epblas-parallel/version.h.in" \
    > "$DOC_DIR/_doxygen/include/epblas-parallel/version.h"

echo ">> doxygen (header -> XML)"
( cd "$DOC_DIR" && doxygen Doxyfile )

echo ">> sphinx-build (-> HTML)"
"$SPHINX" -b html "$DOC_DIR" "$DOC_DIR/_build/html" "$@"

echo ">> done: $DOC_DIR/_build/html/index.html"
