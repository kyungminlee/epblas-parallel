# epblas-openblas — OpenBLAS D/Z port to kind10 (REAL/COMPLEX(KIND=10))

Direct ports of OpenBLAS double-precision (D) real and double-complex (Z)
sources, retyped to 80-bit long double / `_Complex long double`. Builds a
separate static archive `${LIB_PREFIX}blas_openblas` (not wired into the
public `epblas-parallel::${LIB_PREFIX}blas` composite). Tested standalone
against the migrated baseline via `tests/epblas-openblas/`.

> **kind16 (`__float128`) sibling.** `src/epblas-openblas/kind16/` is a
> faithful hand-port of this kind10 tree — same algorithm, threading, and
> thresholds, differing only by the mechanical quad substitution set
> (`long double`→`__float128`, `_Complex long double`→`__complex128`,
> literal `L`→`Q`, zero imaginary literals → `(__complex128)` casts,
> `sqrtl/ldexpl/conjl/creall`→`*q`, `fabsl`→`__builtin_fabsf128`,
> `<quadmath.h>` added alongside the kept `<math.h>`, prefixes
> `e→q`/`y→x`/`ie→iq`/`iy→ix`/`ey→qx`/`ye→xq`). All **75 routines** are
> ported and pass the consistency suite **75/75** against the
> `eplinalg::qblas` migrated baseline (`ctest -R
> 'epblas_openblas_fuzz_(q|x|iq|ix)'`). The two trees are kept in sync by
> hand — see `docs/adr/0002-openblas-kind16-faithful-handport.md`. The
> per-routine table below is kind10-specific (bench numbers not re-gathered
> for kind16; the algorithm is identical at both precisions).

Naming map (project convention, matches `blas/src/`):
- `d*`  (double real)             → `e*`
- `z*`  (double complex)          → `y*`
- `dz*` (real-from-complex)       → `ey*`
- `iz*` (integer-from-complex)    → `iy*`
- `zd*` (complex scaled by real)  → `ye*`

OpenBLAS source roots:
- `external/openblas/interface/<name>.c` — dispatch / argument check / shape
- `external/openblas/kernel/generic/<name>.c` — portable C kernel (no asm)
- `external/openblas/kernel/x86_64/d<name>.c` — D-specific kernel (some have
  microkernel SIMD; ignore — kind10 has no SSE/AVX path)
- `external/openblas/driver/level3/level3.c` — GotoBLAS three-level blocking
  driver for GEMM family (templated by data type via macros)

Per row:
- **status** — `todo` / `wip` / `done`
- **smoke** — fuzz passes, bench builds: `n` / `y`
- **bench-omp1**, **bench-omp4** — most recent GFLOPS overlay vs migrated
- **par>ep (omp1/omp4)** — peak `epblas-parallel / epblas-openblas` overlay ratio at OMP=1 and OMP=4, from the 4-variant sweep in `reports/cmp5/cmp5.tsv`. `—` means within the 10% noise floor at that OMP level (epblas-parallel not meaningfully faster). A value like `1.51× / 2.15×` is the worst (key, size) row for that routine — i.e. the (key, size) at which epblas-parallel most outperforms this epblas-openblas port. `n/d` = no bench data (timed out). `n/a` = no comparable bench (algorithm diverges from OpenBLAS, e.g. Blue's nrm2, or scalar O(1)).

**`faithful` column:**
- `yes` — algorithm + threading mirror OpenBLAS (interface + kernel + thread driver structure all ported; OMP replaces blas_queue but matches the partitioning/reduction shape).
- `partial` — algorithm matches OpenBLAS but threading differs, in either of two ways:
  (a) overlay uses a simpler OMP partition rather than the custom load-balanced width formula from a `*_thread.c` driver, OR
  (b) OpenBLAS is serial at the interface layer (no L1 thread driver, or `#undef SMP` on x86_64) and the overlay adds an OMP block-partition — intentional divergence motivated by the HPC many-core target.
- `no` — diverges from OpenBLAS (e.g., a different algorithm such as Blue's nrm2, or serial when OpenBLAS uses a thread driver).
- `n/a` — scalar / O(1) / no SMP path exists in OpenBLAS for this routine.

## Level 1 — real

| target  | openblas source        | status | smoke | faithful | bench-omp1            | bench-omp4            | par>ep (omp1/omp4)    | notes                              |
|---------|------------------------|--------|-------|----------|-----------------------|-----------------------|-----------------------|------------------------------------|
| eaxpy   | interface/axpy.c       | done   | y     | yes      | 1.40/1.31 (×1.07)     | 5.14/1.31 (×3.94)     | — / 2.36×             | OMP block-partition (mirrors blas_level1_thread); cutoff matches OpenBLAS n>10000 |
| ecopy   | interface/copy.c       | done   | y     | partial  | 26.43/26.32 (×1.00)   | 92.11/25.32 (×3.64)   | 8.55× / 8.85×         | algo matches COPY_K; OMP added (OpenBLAS interface serial) |
| eswap   | interface/swap.c       | done   | y     | partial  | 24.87/25.99 (×0.96)   | 90.43/25.59 (×3.53)   | 1.16× / 1.96×         | algo matches; OMP added (OpenBLAS x86_64 `#undef SMP`) |
| escal   | interface/scal.c       | done   | y     | yes      | 0.82/0.80 (×1.01)     | 2.92/0.79 (×3.69)     | 1.15× / 2.73×         | OMP block-partition (mirrors blas_level1_thread); cutoff n>10000 vs OpenBLAS n>1048576 |
| edot    | interface/dot.c        | done   | y     | partial  | 2.33/2.19 (×1.06)     | 7.38/1.95 (×3.78)     | 1.13× / 3.39×         | 5-way unroll 1-acc (NETLIB DDOT, not OpenBLAS 4-way 4-acc); OMP added (interface serial) |
| enrm2   | interface/nrm2.c       | done   | y     | **no**   | n/a (Blue scl)        | n/a (Blue scl)        | n/a                   | Blue's algorithm — diverges from OpenBLAS's scaled-SSQ |
| easum   | interface/asum.c       | done   | y     | partial  | 1.97/1.42 (×1.39)     | 5.77/1.42 (×4.07)     | 1.61× / 5.09×         | fabsl + 6-acc (NETLIB shape, not OpenBLAS kernel); OMP added (interface serial) |
| erot    | interface/rot.c        | done   | y     | partial  | 1.96/1.83 (×1.07)     | 6.36/1.77 (×3.60)     | — / 1.63×             | algo matches ROT_K; OMP added (OpenBLAS interface serial) |
| erotg   | interface/rotg.c       | done   | y     | n/a      | 0.50/0.46 (×1.09)     | 0.12/0.12 (×0.97)     | 1.19× / 1.15×         | scalar; OpenBLAS has no SMP path   |
| erotm   | interface/rotm.c       | done   | y     | partial  | 1.31/1.32 (×1.00)     | 4.27/1.32 (×3.25)     | 1.21× / 1.84×         | algo matches (dflag-unswitched); OMP added (OpenBLAS interface serial) |
| erotmg  | interface/rotmg.c      | done   | y     | n/a      | 0.48/0.59 (×0.82)     | 0.68/0.53 (×1.29)     | 3.48× / —             | scalar; no SMP                     |
| ieamax  | interface/imax.c       | done   | y     | partial  | 0.60/0.53 (×1.12)     | 2.65/0.52 (×5.09)     | 2.10× / 4.56×         | algo matches IDAMAX_K (first-wins); OMP added (OpenBLAS interface serial) |

## Level 1 — complex

| target  | openblas source        | status | smoke | faithful | bench-omp1            | bench-omp4            | par>ep (omp1/omp4)    | notes                              |
|---------|------------------------|--------|-------|----------|-----------------------|-----------------------|-----------------------|------------------------------------|
| yaxpy   | interface/axpy.c       | done   | y     | yes      | 2.00/2.00 (×1.00)     | 7.78/2.01 (×3.87)     | — / 1.59×             | OMP block-partition (mirrors blas_level1_thread); cutoff matches OpenBLAS n>10000 |
| ycopy   | interface/copy.c       | done   | y     | partial  | 24.43/24.64 (×0.99)   | 93.82/25.51 (×3.68)   | 8.87× / 8.88×         | algo matches COPY_K; OMP added (OpenBLAS interface serial) |
| yswap   | interface/swap.c       | done   | y     | partial  | 25.00/24.37 (×1.03)   | 76.73/23.71 (×3.24)   | — / 1.54×             | algo matches; OMP added (OpenBLAS x86_64 `#undef SMP`) |
| yscal   | interface/scal.c       | done   | y     | yes      | 2.00/1.87 (×1.07)     | 7.11/1.97 (×3.61)     | — / 1.76×             | manual re/im, xi*ar first (-1 fxch); cutoff n>10000 vs OpenBLAS n>1048576 |
| yescal  | interface/scal.c       | done   | y     | yes      | 2.18/2.03 (×1.07)     | 8.01/2.11 (×3.79)     | — / 1.97×             | explicit 2-fmul pair walk; cutoff n>10000 vs OpenBLAS n>1048576 |
| ydotu   | interface/dot.c        | done   | y     | partial  | 3.16/3.11 (×1.02)     | 10.97/3.03 (×3.62)    | 1.61× / 1.82×         | algo matches DOTU_K; OMP added (OpenBLAS interface serial) |
| ydotc   | interface/dot.c        | done   | y     | partial  | 2.79/2.77 (×1.01)     | 10.72/2.79 (×3.85)    | — / 1.76×             | algo matches DOTC_K (conjg x); OMP added (OpenBLAS interface serial) |
| eynrm2  | interface/nrm2.c       | done   | y     | **no**   | n/a (Blue scl)        | n/a (Blue scl)        | n/a                   | Blue's algorithm — diverges from OpenBLAS |
| eyasum  | interface/asum.c       | done   | y     | partial  | 1.24/0.78 (×1.59)     | 4.66/0.78 (×6.00)     | 1.68× / 3.48×         |Re\|+\|Im\| 1.92× / 1.64×         |); OMP added (OpenBLAS interface serial) |
| yerot   | interface/rot.c (zrot) | done   | y     | partial  | 1.96/1.01 (×1.94)     | 6.79/0.99 (×6.84)     | — / 1.60×             | algo matches ROT_K (zrot variant); OMP added (OpenBLAS interface serial) |
| yrotg   | interface/rotg.c       | done   | y     | n/a      | 0.05/0.05 (×0.96)     | 0.05/0.05 (×0.97)     | 12.26× / 12.18×       | scalar; no SMP                     |
| iyamax  | interface/imax.c       | done   | y     | partial  | 1.15/0.95 (×1.22)     | 4.06/0.93 (×4.37)     | 1.15× / 2.43×         | algo matches IZAMAX_K (first-wins); OMP added (OpenBLAS interface serial) |
| ecabs1  | kernel/generic/cabs.c  | done   | y     | n/a      | n/a (scalar)          | n/a (scalar)          | n/a                   | scalar O(1)                        |

## Level 2 — real  (bench at N=512, kernel-only, unit strides)

| target  | openblas source        | status | smoke | faithful | bench-omp1            | bench-omp4            | par>ep (omp1/omp4)    | notes |
|---------|------------------------|--------|-------|----------|-----------------------|-----------------------|-----------------------|-------|
| egemv   | interface/gemv.c       | done   | y     | yes      | 1.69/1.88 (×0.90)     | 5.67/1.83 (×3.09)     | 1.80× / 3.65×         | M-split + gemv_thread split_x fallback (m < 4·nthreads, m·n large) |
| eger    | interface/ger.c        | done   | y     | yes      | 1.08/1.10 (×0.99)     | 4.72/1.20 (×3.94)     | — / 2.40×             | N-block partition + ger_thread x-buffer-copy for incx≠1 |
| esymv   | interface/symv.c       | done   | y     | yes      | 1.73/1.96 (×0.88)     | 6.59/1.84 (×3.57)     | 1.17× / 1.19×         | symv_thread sqrt-partition (mask=3 min-4) + AXPY-chain reduce |
| esyr    | interface/syr.c        | done   | y     | yes      | 1.01/1.00 (×1.01)     | 2.76/0.99 (×2.79)     | — / 1.12×             | syr_thread sqrt-partition (mask=7 min-16); UPPER reverse-mapped |
| esyr2   | interface/syr2.c       | done   | y     | yes      | 1.85/1.87 (×0.99)     | 6.28/1.67 (×3.76)     | 1.11× / —             | syr2_thread sqrt-partition (mask=7 min-16); UPPER reverse-mapped |
| espr    | interface/spr.c        | done   | y     | yes      | 1.20/1.20 (×1.00)     | 3.95/1.05 (×3.74)     | —                     | spr_thread sqrt-partition (mask=7 min-16); packed col offsets |
| espr2   | interface/spr2.c       | done   | y     | yes      | 0.95/1.85 (×0.51)     | 6.31/1.68 (×3.75)     | 1.11× / —             | spr2_thread sqrt-partition (mask=7 min-16); packed col offsets |
| espmv   | interface/spmv.c       | done   | y     | yes      | 1.69/1.87 (×0.90)     | 6.73/1.87 (×3.60)     | 1.36× / 1.63×         | sqrt-partition (mask=3 min-4, symv_thread shape — not spmv_thread's mask=7 min-16) + AXPY-chain reduce; packed col offsets |
| esbmv   | interface/sbmv.c       | done   | y     | yes      | 1.71/1.69 (×1.01)     | 1.71/1.69 (×1.01)     | — / 4.82×             | sbmv_thread port: load-balanced cols + y_priv reduce + alpha-AXPY |
| egbmv   | interface/gbmv.c       | done   | y     | yes      | 1.00/1.16 (×0.86)     | 1.02/1.16 (×0.88)     | 1.22× / 3.46×         | gbmv_thread port: col-partition + y_priv reduce + alpha-AXPY |
| etrmv   | interface/trmv.c       | done   | y     | yes      | 1.20/1.03 (×1.17)     | 1.21/1.05 (×1.15)     | 1.48× / 3.09×         | trmv_thread port: sqrt-partition + DTB_ENTRIES tile + per-thread slot + AXPY-reduce |
| etrsv   | interface/trsv.c       | done   | y     | yes      | 1.19/1.22 (×0.97)     | 1.20/1.20 (×1.00)     | 1.59× / 1.57×         | serial — matches OpenBLAS (no trsv_thread.c) |
| etbmv   | interface/tbmv.c       | done   | y     | yes      | 0.98/1.12 (×0.88)     | 0.98/1.06 (×0.92)     | 1.23× / 11.89×        | tbmv_thread port: sqrt/even partition + private slots + full AXPY-reduce |
| etbsv   | interface/tbsv.c       | done   | y     | yes      | 0.92/1.00 (×0.92)     | 0.97/1.01 (×0.96)     | 1.45× / 1.44×         | serial — matches OpenBLAS (no tbsv_thread.c) |
| etpmv   | interface/tpmv.c       | done   | y     | yes      | 1.02/1.05 (×0.97)     | 1.02/1.04 (×0.98)     | — / 2.86×             | tpmv_thread port: sqrt-partition + per-thread slot + AXPY-reduce |
| etpsv   | interface/tpsv.c       | done   | y     | yes      | 1.18/1.14 (×1.03)     | 1.20/1.20 (×1.00)     | 1.18× / 1.11×         | serial — matches OpenBLAS (no tpsv_thread.c) |

## Level 2 — complex (general / sym / triangular / band / packed)

| target  | openblas source        | status | smoke | faithful | bench-omp1            | bench-omp4            | par>ep (omp1/omp4)    | notes |
|---------|------------------------|--------|-------|----------|-----------------------|-----------------------|-----------------------|-------|
| ygemv   | interface/gemv.c       | done   | y     | yes      | 1.77/2.67 (×0.66)     | 6.76/2.61 (×2.59)     | 1.46× / 1.43×         | M-split + gemv_thread split_x fallback (m < 4·nthreads, m·n large) |
| ygeru   | interface/ger.c (zgeru)| done   | y     | yes      | 1.89/1.90 (×1.00)     | 7.50/1.89 (×3.97)     | — / 1.65×             | N-block partition + ger_thread x-buffer-copy for incx≠1 |
| ygerc   | interface/ger.c (zgerc)| done   | y     | yes      | 1.84/1.80 (×1.02)     | 7.44/1.89 (×3.92)     | — / 1.69×             | N-block partition + conjg y; same shape as ygeru |
| ygbmv   | interface/zgbmv.c      | done   | y     | yes      | 1.94/1.95 (×1.00)     | 1.85/1.95 (×0.95)     | 1.13× / 1.88×         | gbmv_thread port: col-partition + y_priv reduce + alpha-AXPY (N/T/C) |
| ytrmv   | interface/trmv.c       | done   | y     | yes      | 1.96/1.97 (×0.99)     | 1.97/1.89 (×1.04)     | — / 1.78×             | trmv_thread port: sqrt-partition + DTB_ENTRIES tile + per-thread slot + AXPY-reduce (N/T/C) |
| ytrsv   | interface/trsv.c       | done   | y     | yes      | 0.55/0.53 (×1.03)     | 0.54/0.55 (×0.98)     | n/d                   | serial — matches OpenBLAS |
| ytbmv   | interface/tbmv.c       | done   | y     | yes      | 1.96/1.87 (×1.05)     | 1.88/1.86 (×1.01)     | 1.39× / 5.35×         | tbmv_thread port: sqrt/even partition + private slots + full AXPY-reduce (N/T/C) |
| ytbsv   | interface/tbsv.c       | done   | y     | yes      | 0.06/0.06 (×1.01)     | 0.06/0.06 (×0.99)     | 1.32× / 1.19×         | serial — matches OpenBLAS |
| ytpmv   | interface/tpmv.c       | done   | y     | yes      | 1.99/2.04 (×0.97)     | 2.01/2.02 (×0.99)     | 1.27× / 1.72×         | tpmv_thread port: sqrt-partition + per-thread slot + AXPY-reduce (N/T/C) |
| ytpsv   | interface/tpsv.c       | done   | y     | yes      | 0.56/0.58 (×0.97)     | 0.56/0.57 (×0.98)     | n/d                   | serial — matches OpenBLAS |

## Level 2 — complex Hermitian

| target  | openblas source        | status | smoke | faithful | bench-omp1            | bench-omp4            | par>ep (omp1/omp4)    | notes |
|---------|------------------------|--------|-------|----------|-----------------------|-----------------------|-----------------------|-------|
| yhemv   | interface/zhemv.c      | done   | y     | yes      | 2.66/1.99 (×1.34)     | 10.36/2.33 (×4.45)    | 1.12× / 1.12×         | symv_thread+HEMV sqrt-partition + AXPY-chain reduce; conj A^H reflection |
| yhbmv   | interface/zhbmv.c      | done   | y     | yes      | 2.56/2.01 (×1.27)     | 2.58/2.01 (×1.28)     | 1.11× / 2.51×         | sbmv_thread+HEMV port: DOTC band + REAL diagonal |
| yher    | interface/zher.c       | done   | y     | yes      | 1.69/1.69 (×1.00)     | 6.89/1.87 (×3.68)     | —                     | syr_thread+HEMV sqrt-partition (mask=7 min-16); alpha REAL |
| yher2   | interface/zher2.c      | done   | y     | yes      | 2.46/2.23 (×1.10)     | 9.69/2.76 (×3.51)     | —                     | syr2_thread+HEMV sqrt-partition (mask=7 min-16) |
| yhpmv   | interface/zhpmv.c      | done   | y     | yes      | 2.66/1.98 (×1.34)     | 5.55/1.80 (×3.09)     | 1.18× / 1.23×         | sqrt-partition (mask=3 min-4, symv_thread shape — not spmv_thread's mask=7 min-16) + HEMV AXPY-chain reduce; packed |
| yhpr    | interface/zhpr.c       | done   | y     | yes      | 1.95/1.95 (×1.00)     | 7.09/1.86 (×3.81)     | 1.10× / —             | spr_thread+HEMV sqrt-partition (mask=7 min-16); packed |
| yhpr2   | interface/zhpr2.c      | done   | y     | yes      | 2.79/2.46 (×1.13)     | 10.52/2.92 (×3.61)    | —                     | spr2_thread+HEMV sqrt-partition (mask=7 min-16); packed |

## Level 3 — real

| target  | openblas source        | status | smoke | faithful | bench-omp1 | bench-omp4 | par>ep (omp1/omp4)    | notes |
|---------|------------------------|--------|-------|----------|------------|------------|-----------------------|-------|
| egemm   | interface/gemm.c       | done   | y     | partial  | 3.08/1.88 (×1.64)     | 11.26/1.86 (×6.04)    | — / 1.49×             | OpenBLAS gemmkernel_2x2 + ncopy/tcopy_2 packers + gemm_beta; per-thread Ap, shared Bp packed under omp single + omp barrier; M-axis OMP block-partition (OpenBLAS level3.c structure + OMP replaces blas_queue) |
| egemmtr | interface/gemmt.c      | done   | y     | yes      | 1.44/1.51 (×0.95)     | 3.05/1.55 (×1.97)     | 2.92× / 5.78×         | Direct port of OpenBLAS interface/gemmt.c — column-by-column GEMV recipe (no L3 blocking nest exists in OpenBLAS for GEMMT). For each output column i: SCAL the active C-slice by beta inline (len = j = n-i for LOWER, j = i+1 for UPPER, stride 1), then one egemv_ call into that slice (TRANS=ta, M/N flipped per ta, x = i-th col of op(B), incx = 1 if tb='N' else ldb). egemv_ provides the SMP layer internally; column loop is sequential (matches OpenBLAS's per-column gemv_thread dispatch, no nested OMP). Note: transb='T' configs serialize at OMP=4 because egemv 'N' path with incx≠1 falls to the scalar non-unit-stride branch; transb='N' configs scale fully (×2.0–2.5 in those rows) |
| esymm   | interface/symm.c       | done   | y     | partial  | 2.87/1.46 (×1.96)     | 8.48/1.47 (×5.77)     | —                     | OpenBLAS symm_{u,l}copy_2 packers + side-swap (SIDE=R puts the symm matrix in B's slot internally so OCOPY is the SYMM packer); reuses eblas_egemm_kernel + ncopy/tcopy + beta from common/eblas_l3_real.c; per-thread Ap, shared Bp packed under omp single + omp barrier; M-axis OMP block-partition (mirrors egemm structure) |
| esyrk   | interface/syrk.c       | done   | y     | partial  | 2.85/1.62 (×1.76)     | 5.35/1.57 (×3.41)     | —                     | OpenBLAS level3_syrk.c + syrk_kernel.c — diagonal-aware kernel (eblas_esyrk_kernel_{u,l}) splits each tile into pre-diag/diag/post-diag sub-GEMMs; diag block GEMM'd into a small NR×NR subbuf and mask-merged via UPLO triangle; triangular β pre-pass (eblas_esyrk_beta_{u,l}) leaves off-UPLO untouched; reuses eblas_egemm_kernel + ncopy/tcopy from common/eblas_l3_real.c (both Ap and Bp source from A); per-thread Ap, shared Bp; M-axis OMP block-partition with per-(js) UPLO clip (mirrors level3_syrk.c m_start/m_end) |
| esyr2k  | interface/syr2k.c      | done   | y     | partial  | 2.72/2.17 (×1.25)     | 5.26/2.16 (×2.43)     | 1.11× / 1.52×         | OpenBLAS level3_syr2k.c + syr2k_kernel.c — two-pass per (is, js) tile: pass 1 calls eblas_esyr2k_kernel_{u,l} with (Ap=A-pack, Bp=B-pack, flag=1) writing alpha*A*B^T into kept-triangle strips AND merging the diagonal NR×NR block via symmetric mirror subbuf[i,j]+subbuf[j,i] (captures both A*B^T and B*A^T in one shot); pass 2 with (Ap=B-pack, Bp=A-pack, flag=0) writes alpha*B*A^T into kept-triangle strips only, skipping the diagonal block; reuses eblas_egemm_kernel + ncopy/tcopy from common/eblas_l3_real.c; triangular β pre-pass reuses eblas_esyrk_beta_{u,l} (off-UPLO untouched); two B-side packs (Bp_A, Bp_B) shared across all M-strips, two per-thread A-side packs (Ap_A, Ap_B); M-axis OMP block-partition with per-(js) UPLO clip (mirrors level3_syr2k.c m_start/m_end) |
| etrmm   | interface/trsm.c (TRMM macro) | done | y | partial  | 2.82/1.31 (×2.15)     | 6.50/1.29 (×5.05)     | — / 1.62×             | Full OpenBLAS L3 port: trmm_L.c / trmm_R.c blocking nest (NC×KC×MC), pack-and-conquer with 4 TRMM A-side packers (eblas_etrmm_i{ut,un,lt,ln}copy) + diagonal-aware TRMM kernel (eblas_etrmm_kernel collapsing the 4 (LEFT,TRANSA) variants under runtime flags) + GEMM kernel for off-diagonal sub-tiles (eblas_egemm_kernel_store overwrite variant). Same source packer serves both SIDE=L (I-role) and SIDE=R (O-role) — but the (uplo,trans)→packer mapping differs by SIDE (UT↔UN, LT↔LN swap). Alpha pre-scales B once, then kernel runs with alpha=1 (matches OpenBLAS GEMM_BETA + GEMM_KERNEL(beta=0) convention). OMP: SIDE=L partitions N-axis (one js-range per thread); SIDE=R partitions M-axis. Each thread owns its own Ap and Bp scratch (no cross-thread sync) |
| etrsm   | interface/trsm.c       | done   | y     | partial  | 2.82/1.20 (×2.34)     | 10.00/1.29 (×7.78)    | 2.97× / 2.92×         | Full OpenBLAS L3 port: trsm_L.c / trsm_R.c blocking nest (NC×KC×MC), pack-and-conquer with 4 TRSM A-side packers (eblas_etrsm_i{ut,un,lt,ln}copy — diagonal-aware, bake `inv(diag(A))` into packed buffer) + collapsed diagonal-aware TRSM kernel (eblas_etrsm_kernel dispatching the 4 (LEFT,TRANSA) OpenBLAS variants LN/LT/RN/RT via runtime flags; per-variant solve() function + per-variant outer (i,j) walk). Off-diagonal sub-tiles use eblas_egemm_kernel with α=dm1=-1 (+= semantics matches OpenBLAS GEMM_KERNEL inside the TRSM kernel). Alpha pre-scales B once (matches OpenBLAS args.beta=alpha pre-pass). OMP: SIDE=L partitions N-axis (gemm_thread_n shape); SIDE=R partitions M-axis (gemm_thread_m). Per-thread Ap+Bp (no cross-thread sync) |

## Level 3 — complex (general / sym / triangular)

| target  | openblas source        | status | smoke | faithful | bench-omp1 | bench-omp4 | par>ep (omp1/omp4)    | notes |
|---------|------------------------|--------|-------|----------|------------|------------|-----------------------|-------|
| ygemm   | interface/gemm.c       | done   | y     | partial  | 2.23/2.42 (×0.92)     | 7.94/2.40 (×3.31)     | 1.36× / 1.41×         | OpenBLAS zgemmkernel_2x2 + zgemm_{ncopy,tcopy}_2 packers + zgemm_beta; conjugation absorbed into the packers so the kernel only runs the NN path (1 kernel covers all 16 transa/transb codes); per-thread Ap, shared Bp packed under omp single + omp barrier; M-axis OMP block-partition (mirrors egemm structure) |
| ygemmtr | interface/gemmt.c      | done   | y     | yes      | 2.02/2.17 (×0.93)     | 3.17/2.30 (×1.38)     | 1.36× / 2.20×         | Complex twin of egemmtr — same column-by-column GEMV recipe via ygemv_. Conjugation handled per OpenBLAS interface/gemmt.c: transa/transb codes 'R' (ConjNoTrans) trigger in-place A/B conjugation around the loop (IMATCOPY_K_CNC equivalent — restored after); 'C' (ConjTrans) passes through to ygemv 'C' directly. Same per-column SCAL inline + ygemv call as the real path. Same OMP=4 limitation as egemmtr: transb in {T,C} configs serialize because ygemv 'N' path needs incx==1 to enter the parallel branch |
| ysymm   | interface/symm.c       | done   | y     | partial  | 2.13/2.20 (×0.97)     | 7.82/2.10 (×3.72)     | 1.21× / 1.29×         | OpenBLAS zsymm_{u,l}copy_2 packers + side-swap (SIDE=R puts the symm matrix in B's slot internally); reuses eblas_ygemm_kernel + ncopy/tcopy + beta from common/eblas_l3_complex.c; SYMM packer copies (re,im) through unchanged (no conjugation — that's HEMM); per-thread Ap, shared Bp packed under omp single + omp barrier; M-axis OMP block-partition (mirrors ygemm structure) |
| ysyrk   | interface/syrk.c       | done   | y     | partial  | 2.15/2.43 (×0.88)     | 4.77/2.41 (×1.98)     | 1.47× / 1.55×         | OpenBLAS level3_syrk.c + syrk_kernel.c — same diagonal-split pattern as esyrk, but per-element width = 2 long doubles (interleaved re,im); SYMM has no conjugation so packers called with conj=0 (HERK is yherk, #65); reuses eblas_ygemm_kernel + ncopy/tcopy from common/eblas_l3_complex.c; triangular β pre-pass leaves off-UPLO untouched; OMP=1 trans='T' cases run sub-parity (×0.66–0.77) — same pattern as ygemm OMP=1 ×0.92, complex L3 packing overhead not fully amortized on small problems |
| ysyr2k  | interface/syr2k.c      | done   | y     | partial  | 2.14/2.52 (×0.85)     | 4.33/2.49 (×1.74)     | 1.38× / 1.54×         | OpenBLAS level3_syr2k.c + syr2k_kernel.c (Z-variant) — same two-pass diagonal-merge structure as esyr2k, with per-element width = 2 long doubles (interleaved re,im); SYR2K has no conjugation so packers called with conj=0 (HER2K is yher2k, #66); reuses eblas_ygemm_kernel + ncopy/tcopy from common/eblas_l3_complex.c; triangular β pre-pass reuses eblas_ysyrk_beta_{u,l}; OMP=1 cases run sub-parity (×0.71–0.92) — same complex-L3 packing-overhead pattern as ygemm/ysyrk OMP=1 |
| ytrmm   | interface/trsm.c (TRMM macro) | done | y | partial  | 2.05/2.26 (×0.91)     | 6.39/2.24 (×2.85)     | 1.60× / 2.06×         | Full OpenBLAS L3 port (complex twin of etrmm): trmm_L.c / trmm_R.c blocking nest with 4 complex TRMM A-side packers (eblas_ytrmm_i{ut,un,lt,ln}copy) and diagonal-aware kernel (eblas_ytrmm_kernel, NN-only path — conjugation absorbed by the packers via the `conj` flag, matching ygemm pattern). Off-diagonal sub-tiles use eblas_ygemm_kernel_store (overwrite variant). Same SIDE=L N-axis / SIDE=R M-axis OMP partitioning; per-thread Ap+Bp (no cross-thread sync). OMP=1 sub-parity (×0.91) follows the complex-L3 packing-overhead pattern shared with ygemm/ysyrk/ysyr2k at OMP=1; OMP=4 recovers solid parallel speedup (×2.85) |
| ytrsm   | interface/trsm.c       | done   | y     | partial  | 2.24/2.23 (×1.00)     | 8.18/2.23 (×3.68)     | 1.80× / 2.03×         | Full OpenBLAS L3 port (complex twin of etrsm): trsm_L.c / trsm_R.c blocking nest with 4 complex TRSM A-side packers (eblas_ytrsm_i{ut,un,lt,ln}copy — embed compinv(diag) via Smith's reciprocal, sign-flip imag at pack time when conj) and collapsed diagonal-aware TRSM kernel (eblas_ytrsm_kernel; NN-only complex multiply — conjugation absorbed by packers via the `conj` flag, matching ygemm pattern). Off-diagonal sub-tiles use eblas_ygemm_kernel with α=(-1+0i). Same SIDE=L N-axis / SIDE=R M-axis OMP partitioning as etrsm; per-thread Ap+Bp. OMP=1 parity (×1.00) — complex L3 packing overhead amortized exactly across N=512 ratio; OMP=4 recovers a solid ×3.68 |

## Level 3 — complex Hermitian

| target  | openblas source        | status | smoke | faithful | bench-omp1 | bench-omp4 | par>ep (omp1/omp4)    | notes |
|---------|------------------------|--------|-------|----------|------------|------------|-----------------------|-------|
| yhemm   | interface/symm.c (HEMM macro) | done | y | partial  | 2.19/2.15 (×1.02)     | 5.74/2.23 (×2.58)     | 1.20× / 1.29×         | Full OpenBLAS L3 port: zhemm_k.c blocking nest + side-swap (SIDE=R puts the Hermitian matrix in B's slot internally, same as ysymm). Hermitian-aware packers (eblas_yhemm_{u,l}copy) port zhemm_{u,l}tcopy_2.c — negate imag on the reflected-across-diagonal half, zero the diagonal imag (Hermitian diagonals are real by definition). OpenBLAS shares one packer source between ICOPY (SIDE=L) and OCOPY (SIDE=R) by compiling zhemm_k.c with -DNC for RSIDE → switches the kernel to GEMM_KERNEL_R, which conjugates Bp once more during the multiply, cancelling the extra conjugation the IC-style packer baked in. Our shared kernel is NN-only so we ship explicit OC variants (eblas_yhemm_{u,l}copy_oc) with the imag-sign branches inverted, used for SIDE=R only. M-axis OMP block-partition; per-thread Ap, shared Bp packed under omp single + omp barrier (same shape as ysymm/ygemm) |
| yherk   | interface/syrk.c (HEMM macro) | done | y | partial  | 2.25/2.34 (×0.96)     | 4.57/2.31 (×1.98)     | 1.43× / 1.34×         | Full OpenBLAS L3 port: zherk_k.c routes through level3_syrk.c with the kernel swapped for zherk_kernel.c — UPLO-aware diagonal-tile writeback that forces imag = 0 on the diagonal (Hermitian C contract). eblas_yherk_kernel_{u,l} is the SYRK twin with that one writeback change; eblas_yherk_beta_{u,l} ports zherk_beta.c (real beta, unconditional diag imag = 0 even when β=1). Conjugation absorbed at pack time: upstream picks GEMM_KERNEL_R for TRANS='N' (conj Bp) and GEMM_KERNEL_L for TRANS='C' (conj Ap); our shared NN kernel reuses ygemm_{n,t}copy with conj=1 on the matching side instead of growing a 2nd kernel variant. M-axis OMP block-partition with UPLO clip (same shape as ysyrk); per-thread Ap, shared Bp packed under omp single + omp barrier. OMP=1 TRANS='C' loses ~25% to migrated Fortran ZHERK (its dotc-style inner loop vectorizes well), parity on TRANS='N'; OMP=4 wins across all 4 SU configs |
| yher2k  | interface/syr2k.c (HEMM macro) | done | y | partial  | 2.25/2.20 (×1.02)     | 4.54/2.20 (×2.06)     | 1.25× / 1.39×         | Full OpenBLAS L3 port: zher2k_k.c routes through level3_syr2k.c with the kernel swapped for zher2k_kernel.c — UPLO-aware diagonal subblock writeback that mirrors α·A·B^H's conjugate-partner (subbuf[i,j].im − subbuf[j,i].im) and forces imag = 0 on the actual diagonal element (Hermitian C contract). eblas_yher2k_kernel_{u,l} is the SYR2K twin with that one writeback change; β pre-pass reuses eblas_yherk_beta_{u,l} (real β + diag imag = 0, identical to zherk_beta.c shape). Two-pass per (jc, pc) tile: pass 1 with (αr, αi) flag=1, pass 2 swapped (Ap=B, Bp=A) with (αr, −αi) flag=0 — matches upstream's KERNEL_OPERATION_C negating ALPHA[1] for the conj(α)·B·A^H pass. Conjugation absorbed at pack time: TRANS='N' (upstream GEMM_KERNEL_R) → conj on Bp side; TRANS='C' (upstream GEMM_KERNEL_L) → conj on Ap side; both passes use the same conj assignment, only A/B swap. M-axis OMP block-partition with UPLO clip (same shape as ysyr2k); per-thread (Ap_A, Ap_B), shared (Bp_A, Bp_B) packed under omp single + omp barrier. OMP=1 reaches parity with migrated Fortran ZHER2K (×1.02); OMP=4 wins across all 16 SU configs (1.46×–2.45×, avg 2.06×) |

(Some Hermitian L3 sources live in `driver/level3/` rather than `interface/`
because the dispatcher hands directly to the GEMM driver with a flag —
verify the actual entry points when porting.)

---

Total: 75 routines (matches `blas/src/` migrated set).

How to update this file as ports land: change `status` to `wip` when starting,
`done` when fuzz passes; flip `smoke` to `y` once the CMake list is wired and
the test driver builds; fill `bench-omp1`/`bench-omp4` with the most recent
median GFLOPS, formatted `ov / mig (×speedup)`, e.g. `12.3 / 8.1 (×1.52)`.
Use `notes` to record kernel-shape decisions worth remembering (algorithm
variant, what was tried and reverted, why a faithful straight port was
swapped out): e.g. `fabsl + 6-acc (was ternary+4-acc)` or `Blue's algorithm
(Algorithm 978)`.

Bench numbers are **kernel-only**: the perf harness measures each BLAS
call individually via `perf_now_s()` and excludes the per-iter memcpy
reset from the timed window. Previously the SCAL/COPY/SWAP family
appeared to cap at ~2× OMP=4 speedup, but that was a serial-memcpy
Amdahl artifact in the harness, not the kernel. The peak GFLOPS columns
above now reflect actual kernel scaling (4 threads on this 6-core box).

Note for the **rot family** (erot, erotm, yerot): the earlier
"DRAM-BW bound" diagnosis (2026-05-22) was a measurement artifact —
when re-benched in a clean stage-e build on the same 6-core
DDR4-2666 box, OMP=4 scales 3.2–3.6× on top of OMP=1 for erot
and erotm at N=65536, and yerot stacks the overlay's ~×1.9 OMP=1
margin with another ×3.5 from the 4 cores. The OMP wiring matters
even on a consumer box, not just for the HPC deployment target.
