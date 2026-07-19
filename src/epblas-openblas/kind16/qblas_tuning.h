/*
 * qblas_tuning.h — shared tuning constants for the kind16 (__float128)
 * hand-ported OpenBLAS comparison leg.
 *
 * Single source of truth for values that were previously restated at the
 * top of every routine file (hoisted verbatim; values unchanged, codegen
 * identical).  All values are inherited unchanged from the kind10 leg's
 * tuning — the two legs share a 16-byte element size, so the byte-based
 * footprints coincide.
 */
#ifndef QBLAS_TUNING_H
#define QBLAS_TUNING_H

/* Stay-serial-below-this thresholds for the gated OMP fast paths.
 *   L1:    element count n     (vector maps / reductions)
 *   L2_NN: n*n work threshold  (square/packed/banded matvec, rank-1/2)
 *   L2_MN: m*n work threshold  (rectangular gemv / ger) */
#define QBLAS_MT_MIN_L1     10000
#define QBLAS_MT_MIN_L2_NN  16384
#define QBLAS_MT_MIN_L2_MN  4096

/* L3 pack-panel footprint target: the adaptive-MC growth for small K aims
 * an MC*KC slab at this many bytes (P-core L2 nominal). */
#define QBLAS_L2_TARGET_BYTES (256L * 1024L)

/* Hard cap on partition slots in the partitioned L2 updates. */
#define MAX_PARTITION_CPUS  256

/* Cap on per-thread partial accumulators in the L1 reductions
 * (partial[]/pidx[]/pmax[] stack arrays and their nthreads clamp). */
#define QBLAS_L1_MAX_THREADS  64

#endif /* QBLAS_TUNING_H */
