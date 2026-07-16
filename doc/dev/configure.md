# Configure

## Toolchain

| Need | For |
|---|---|
| C / C++ compiler with OpenMP | the overlay kernels (`gcc`/`g++` + `libgomp`). |
| Fortran compiler | required at `project()` time even for `-DBUILD_TESTING=OFF`. |
| `libquadmath` | the kind16 (`__float128`) and multifloats targets. |
| `fypp` on `PATH` | test/bench build only — the drivers are `.fypp`-templated. |
| `taskset`, `objcopy`/`nm`/`ar` (binutils) | perf harness only (pinning + archive namespacing). |
| `gfortran-15` / `g++-15` | perf harness only — the multifloats `mig` leg. |

The pinned production compiler is **gcc/gfortran-15**; CI also exercises -12.
See [debugging.md](debugging.md) for why the *build* compiler matters when
chasing codegen.

## Options

Override with `-D<option>=<value>` at configure time.

| Option | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE` | — | set `Release` for tuned kernels. |
| `BUILD_TESTING` | `ON` | build the test/bench suite (needs `fypp` + eplinalg). `OFF` = production-only. |
| `EPBLAS_POSITION_INDEPENDENT_CODE` | `ON` | compile archives `-fPIC`. `OFF` = executable-only static build. |
| `EPBLAS_BUILD_SHARED` | `OFF` | also emit shared `.so` overlays (forces PIC ON). |
| `EPBLAS_PARALLEL_BUILD_multifloats` | `ON` | build the `multifloats` target (fetches `multifloats`); `OFF` skips the fetch. |
| `EPBLAS_PARALLEL_USE_SYSTEM_MULTIFLOATS` | `OFF` | consume `multifloats` via `find_package()` instead of FetchContent (nothing bundled). |
| `EPBLAS_PREFER_IOMP5` | `ON` | use Intel `libiomp5` as the OpenMP runtime when found (GOMP-ABI compatible; hot-team reuse cuts the wakeup tax). |
| `MULTIFLOATS_DIR` | — | local `multifloats` checkout, bypassing the network fetch. |

## Presets

`CMakePresets.json` ships `configure`/`build` presets (`linux`) and four
`cmake --workflow` chains (configure → build → labelled ctest):

| Preset | Runs | Wall-clock |
|---|---|---|
| `fuzz` | the correctness gate | ~15 s |
| `perf` | C `perf_*` harnesses | tens of minutes |
| `sweep` | the dual-link perf sweep per family | hours |
| `e2e` | everything above | overnight |

```bash
CMAKE_PREFIX_PATH=/opt/eplinalg-q cmake --workflow --preset fuzz
```

## eplinalg reference (tests/bench only)

Tests A/B the overlay against eplinalg's migrated baseline, so each target you
want exercised needs an eplinalg install reachable via `CMAKE_PREFIX_PATH`.
Install one per target (`e`→kind10, `q`→kind16, `m`→multifloats):

```bash
# In the eplinalg/ repo:
uv run python -m migrator stage /tmp/stage-q --target kind16
cmake -S /tmp/stage-q -B /tmp/stage-q/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/stage-q/build -j8
cmake --install /tmp/stage-q/build --prefix /opt/eplinalg-q
```

Then configure with the prefixes joined by `;`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/eplinalg-e;/opt/eplinalg-q;/opt/eplinalg-m"
```

The configure log reports which targets are testable; a target with no eplinalg
install configures cleanly but contributes no tests:

```
-- epblas-parallel: building target kind16
-- epblas-parallel: tests enabled for target kind16 (qxblas found)
-- epblas-parallel: eyblas not on CMAKE_PREFIX_PATH — tests for kind10 disabled
```
