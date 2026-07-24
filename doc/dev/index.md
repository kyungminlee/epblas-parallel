# Developer documentation

How to configure, build, test, understand, debug, and release
`epblas-parallel`.

| Doc | Covers |
|---|---|
| [configure.md](configure.md) | CMake options, presets, toolchains, `CMAKE_PREFIX_PATH` / eplinalg. |
| [build.md](build.md) | Build commands, artifacts, the generated test/bench drivers, install. |
| [test.md](test.md) | The fuzz correctness gate, seed reproduction, CI. |
| [benchmark.md](benchmark.md) | Measuring performance: the dual-link harness, the pass/fail bars, how to run it. |
| [architecture.md](architecture.md) | Internal design: the two-package split, the overlay linking model, the per-routine module layout. |
| [debugging.md](debugging.md) | Sanitizers, gdb, and the codebase-specific pitfalls (fp80 spill, loop alignment, TLS/PIC, SIMD dispatch). |
| [release.md](release.md) | The version-bump → tag → publish ritual. |

The **library kernels are hand-written** C/C++ — there is no code generator for
the shipped library. The only generated code is the *test/bench* infrastructure
(fypp templates + a Python bench-driver generator), documented in
[build.md](build.md) and [test.md](test.md).

User-facing docs live in [`../user/`](../user/index.md).

```{toctree}
:hidden:

configure
build
test
benchmark
architecture
debugging
release
```
