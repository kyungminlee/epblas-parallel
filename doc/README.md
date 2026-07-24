# epblas-parallel documentation

- [user/](user/index.md) — using the library: install, link, call.
- [dev/](dev/index.md) — developing epblas-parallel: configure, build, test, debug, benchmark, release.
- [changelog.md](changelog.md) — include-stub pulling the root `CHANGELOG.md` into the rendered site.

## Building the HTML site

Built with **Sphinx + MyST** (Furo theme) plus **Doxygen + Breathe** for the
C API; published to GitHub Pages by `.github/workflows/docs.yml` on pushes
to `main`.

One-time setup (via [uv](https://docs.astral.sh/uv/); `doxygen` must be on
the PATH):

```sh
uv venv doc/.venv
uv pip install --python doc/.venv -r doc/requirements.txt
```

Then:

```sh
./doc/build.sh
```

Output lands in `doc/_build/html/index.html`. `conf.py` and `Doxyfile` are
**generated** from the `.in` templates by `build.sh` (the version string
comes from the root `VERSION` file) and are git-ignored — edit the `.in`
files, not the generated ones.
