# Debugging & bug-fixing

## The loop

1. **Reproduce** with a fixed seed — a failing fuzz seed *is* the repro:
   ```bash
   BLAS_FUZZ_SEED=42 ctest --test-dir build -L fuzz -R <routine>
   ```
   The driver prints the offending dims/args on the first mismatch vs the
   baseline. See [test.md](test.md) for the seed rotation.
2. **Fix** in the routine's `_serial.c` / `_parallel.c` / `_kernel.h`
   ([architecture.md](architecture.md)). Keep the serial path bit-exact with
   the baseline; keep the threaded path bit-exact with the serial path.
3. **Verify** across the seed rotation, then re-time if it was a hot path.

## Sanitizers

No sanitizer preset ships; add flags to a throwaway build. ASan+UBSan catches
the common overlay bugs (heap overflow in a pack arena, OOB in a band/packed
index, `uplo`/`trans` mishandling):

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_PREFIX_PATH=/opt/eplinalg-e
cmake --build build-asan -j8
BLAS_FUZZ_SEED=42 ctest --test-dir build-asan -L fuzz -R <routine>
```

Some overflow bugs only fire on **tall, non-square** shapes (e.g. a pack sized
by the wrong dimension) — a square fuzz case can miss them. Widen the case
range or hand-build a `m ≫ n` repro when ASan is quiet but a shape-dependent
failure is suspected.

## gdb

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=RelWithDebInfo ...
gdb --args ./build-dbg/test/epblas-parallel/fuzz_<routine>
```

Set `OMP_NUM_THREADS=1` first to isolate a correctness bug from a threading
race; if it only reproduces threaded, the bug is in the `_parallel.c`
partitioning, not the kernel.

## Codebase-specific pitfalls

These are the traps that have actually bitten this project. **Always
disassemble the object you built** (`objdump -dl`) before blaming codegen — most
"regressions" are placement, not arithmetic.

- **fp80 register spill (kind10).** The x87 stack is tiny; unrolling,
  aliasing reloads, or pointer-indirect kernels spill the accumulator and
  tank IPC. Symptom: `fld`-heavy inner loop, par slower than a byte-identical
  reference. Prefer register-resident scalar accumulators; unroll only when it
  demonstrably hides latency.
- **`__float128` is opaque calls (kind16).** Every op is a `libquadmath` call
  that clobbers xmm; **do not unroll** — a simple one-element loop wins because
  unrolling spills across call boundaries. (The fp80 unroll intuition is
  inverted here.)
- **Loop-placement alignment.** Instruction-identical loops can differ ~10%
  purely from where the loop head lands relative to a 16/32-byte boundary
  (DSB-vs-MITE, BTB aliasing). Fix is a per-file `-falign-loops=` in that
  target's `CMakeLists.txt` — see the existing pins (`mtbsv`, `qtrmm_serial`,
  the kind10 `-falign-loops=16` group). Flags, not source, are the lever.
- **TLS pack arenas + PIC.** The L3 packers use file-local
  `static __thread` arenas. Under `-fPIC`/shared these move from local-exec to
  local-dynamic TLS; a facade must never declare phantom hidden char-length
  args near such a region (frame corruption). Do **not** add TLS-model flags
  without measuring.
- **SIMD runtime dispatch (multifloats).** AVX2 paths are guarded at runtime
  (`mf_dispatch.h`). A mis-scoped compiler-attribute `pop`/`push` can leak
  `avx2,fma` into baseline functions and `#UD` on pre-AVX2 CPUs. Audit with
  `intel-sde -snb -- fuzz_*` — an objdump grep misses compile-time-gated calls.
- **Unbounded fills → NaN/denormal.** An unbounded `x := A·x` overflows and
  triggers ~100× x87/fp128 slowdown that masquerades as a perf gap; solve/
  triangular drivers force diagonal dominance. If a "gap" is enormous, suspect
  the fill.
- **Stale archive.** Rebuilding a `.a` does **not** relink a consumer perf
  binary — build the consumer target before timing. Identical before/after
  ns ⇒ suspect a stale binary.
- **Build compiler ≠ PATH compiler.** The project pins gcc/gfortran-15;
  disassemble the object produced by the *build* compiler, not whatever `gcc`
  resolves to on `PATH`, or you will chase a schedule that isn't shipping.
