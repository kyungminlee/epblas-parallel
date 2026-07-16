# Release

Versioning is semver on `0.x`: **patch = fix, minor = feature**. The version
lives in the top-level `VERSION` file — the single source of truth that CMake
(`project(... VERSION)`), the installed package config, and the generated
`epblas-parallel/version.h` all read from. Also add a `CHANGELOG.md` entry.

## Ritual

1. Land the change on `develop` (its own commit), green on the fuzz gate.
2. Bump the version in its own commit — edit the `VERSION` file (one line) and
   move the `CHANGELOG.md` "Unreleased" entries under the new version:
   ```
   Bump project version to X.Y.Z for the release   # VERSION + CHANGELOG.md
   ```
3. Fast-forward `main` and tag:
   ```bash
   git checkout main && git merge --ff-only develop
   git tag vX.Y.Z                       # lightweight tag on the bump commit
   git push origin develop main vX.Y.Z
   ```
4. The tag push triggers **`.github/workflows/release.yml`** ("Release
   Libraries"): it builds the overlay per compiler, the gfortran-15 slot runs
   `ctest -L fuzz` as the release correctness signal, and it publishes the
   GitHub release with per-compiler tarballs
   (`epblas-parallel-<compiler>-linux-x86_64.tar.gz`) plus a combined tarball.

## CI / workflows

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` (CI fuzz) | push to `main`/`develop`, PRs | fuzz gate across gcc/gfortran-12 & -15 and the seed rotation. |
| `release.yml` (Release Libraries) | push of a `v*` tag (or `workflow_dispatch`) | build + test-slot + publish tarballs. |

Both jobs carry `timeout-minutes` guards so a stalled runner fails fast instead
of spinning to GitHub's 6 h default. Watch a run:

```bash
gh run watch                              # the most recent run
gh run list --workflow=release.yml        # release history
```

If a run hangs on the ctest step with no local repro on either compiler,
suspect transient CI infra (not code): cancel and re-run before investigating.
