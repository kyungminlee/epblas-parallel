# Bug report (escalate to eplinalg): `mblas` Config omits multifloats-helper transparent deps

**Affected project:** `eplinalg` (the migrated extended-precision BLAS/LAPACK
reference library), repo `~/code/eplinalg`, branch `main`.

**Reported from:** `epblas-parallel`, while wiring the `multifloats`
(double-double, prefix `m`/`w`) fuzz/consistency test gate, which A/Bs the
overlay against eplinalg's migrated reference archive
(`find_package(mblas)` → `mblas_migrated`).

**Severity:** blocks consumption of the `mblas` package entirely
(`find_package(mblas)` can never succeed). Quad (`qblas`) and fp80 (`eblas`)
are unaffected.

---

## Symptom

After staging + installing eplinalg's `multifloats` BLAS to a prefix and
pointing a consumer at it, `find_package(mblas)` fails:

```
CMake Error at CMakeLists.txt (find_package):
  Found package configuration file:
    <prefix>/lib/cmake/mblas/mblasConfig.cmake
  but it set mblas_FOUND to FALSE so package "mblas" is considered to be NOT
  FOUND.  Reason given by package:
  The following imported targets are referenced, but are missing:
  eplinalg::la_constants_mf eplinalg::la_xisnan_mf
```

`eblas`/`qblas` (kind10/kind16) `find_package()` cleanly — only `mblas`
(and any other `NEEDS_MULTIFLOATS` target, e.g. `mlapack`) is broken.

## Reproduction

```sh
cd ~/code/eplinalg
uv run python -m migrator stage /tmp/stage-m --target multifloats --libraries blas
cmake -S /tmp/stage-m -B /tmp/stage-m/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/tmp/eplinalg-m \
      -DMULTIFLOATS_DIR=<local multifloats checkout> -DBUILD_TESTING=OFF
cmake --build /tmp/stage-m/build --target mblas -j8   # `all` needs MPI (BLACS); mblas target alone is enough
cmake --install /tmp/stage-m/build

# now, in any project:
#   find_package(mblas REQUIRED)   # with /tmp/eplinalg-m on CMAKE_PREFIX_PATH
# -> mblas_FOUND = FALSE
```

## Root cause

The generated `mblasConfig.cmake` emits a single transparent dependency,
`find_dependency(eplinalgStdBlas)`, and nothing else. But for
`NEEDS_MULTIFLOATS` targets the precision archive `eplinalg::mblas`
**PUBLIC-links two additional factored-out helper packages**, recorded in
`mblasTargets-*.cmake` as:

```
INTERFACE_LINK_LIBRARIES "eplinalg::la_constants_mf;eplinalg::la_xisnan_mf;eplinalg::blas"
```

Those helpers are installed as their own find_package-able Config packages
(`<prefix>/lib/cmake/la_constants_mf/`, `.../la_xisnan_mf/`), but the
`mblas` Config never `find_dependency()`-loads them, so when the targets
file is `include()`d the two `eplinalg::la_constants_mf` /
`eplinalg::la_xisnan_mf` imported targets are undefined → CMake sets
`mblas_FOUND = FALSE`.

The transparent-dependency machinery only ever covers the standard-precision
archive. Two code sites:

1. **`cmake/CMakeLists.txt`, `_install_library_pair()`** (~line 1351) — the
   dependency list passed to the Config generator is built as:

   ```cmake
   set(_precision_deps "")
   if(TARGET ${lib_name} AND NOT "${lib_name}" STREQUAL "${_precision_target}")
       _std_pkg_name(${lib_name} _std_pkg)
       list(APPEND _precision_deps "${_std_pkg}")      # <-- only eplinalgStdBlas
   endif()
   ...
   fortran_install_library(${_precision_target} ... DEPENDS ${_precision_deps})
   ```

   The MF helpers — PUBLIC-linked a few lines earlier at ~406-410:

   ```cmake
   if(NEEDS_MULTIFLOATS)
       target_link_libraries(${_precision_target} PUBLIC $<BUILD_INTERFACE:multifloatsf>)
       if(TARGET la_constants_mf)
           target_link_libraries(${_precision_target} PUBLIC la_constants_mf)
       endif()
       if(TARGET la_xisnan_mf)
           target_link_libraries(${_precision_target} PUBLIC la_xisnan_mf)
       endif()
   endif()
   ```

   — are never added to `_precision_deps`.

2. **`cmake/FortranCompiler.cmake`, `_deps_block` generation** (~line 405-413)
   faithfully emits one `find_dependency()` per `ARG_DEPENDS` entry — so it is
   correct; it just isn't being handed the helper packages. The adjacent
   comment is a tell: *"a factored-out shared package (the standard-precision
   archive, **currently**)"* — the mechanism was only ever wired for one dep.

(Note `multifloatsf` itself is correctly excluded — it is wrapped in
`$<BUILD_INTERFACE:>` so it never enters the install/export set. Only the
two eplinalg-owned helper packages, `la_constants_mf` / `la_xisnan_mf`, are
missing from the deps list.)

## Proposed fix (in eplinalg)

In `_install_library_pair()`, after the `eplinalgStdBlas` block and before
`fortran_install_library(... DEPENDS ${_precision_deps})`, add the MF helper
packages to the transparent-dependency list when they are linked — they are
installed as standalone Configs, exactly the same shape as `eplinalgStdBlas`:

```cmake
# Multifloats targets PUBLIC-link the la_constants_mf / la_xisnan_mf helper
# archives, each installed as its own Config package. Like eplinalgStdBlas,
# they are transparent deps so consumers only need find_package(mblas).
if(NEEDS_MULTIFLOATS)
    foreach(_mf_helper la_constants_mf la_xisnan_mf)
        if(TARGET ${_mf_helper})
            list(APPEND _precision_deps ${_mf_helper})
        endif()
    endforeach()
endif()
```

This makes the generated `mblasConfig.cmake` self-contained
(`find_dependency(eplinalgStdBlas)` + the two helpers) and `find_package(mblas)`
succeeds with no consumer-side changes. Worth refreshing the stale
"(... currently)" comment in `FortranCompiler.cmake` once the mechanism
generalizes beyond the std archive.

## Local workaround in use (until the eplinalg fix lands)

To unblock the `epblas-parallel` multifloats fuzz gate **without** modifying
eplinalg's source, the two missing `find_dependency()` lines are appended by
hand to the *installed* config in the throwaway staging prefix only:

```
/tmp/eplinalg-m/lib/cmake/mblas/mblasConfig.cmake
    find_dependency(eplinalgStdBlas)
+   find_dependency(la_constants_mf)
+   find_dependency(la_xisnan_mf)
```

This is exactly what the proposed eplinalg fix would generate. It lives only
in `/tmp/eplinalg-m` (a regenerable reference install), touches neither git
repo, and will be obsoleted by the upstream fix.
