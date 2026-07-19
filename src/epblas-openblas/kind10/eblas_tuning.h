/*
 * eblas_tuning.h — shared tuning constants for the kind10 OpenBLAS-port leg.
 *
 * Single home for the threading / blocking thresholds that used to be
 * restated at the top of every source file in this directory. Values are
 * unchanged from the per-file copies they replace; hoisting them here is
 * codegen-neutral.
 *
 * Family overrides: a source file may pre-#define MULTI_THREAD_MINIMAL
 * (or MAX_PARTITION_CPUS) BEFORE including this header to keep a
 * family-tuned value — the packed/symmetric L2 family uses 16384 and the
 * gemv/ger family uses 4096; the defaults below are the L1 family's.
 */
#ifndef EPBLAS_OPENBLAS_KIND10_EBLAS_TUNING_H
#define EPBLAS_OPENBLAS_KIND10_EBLAS_TUNING_H

/* Minimum element count before an L1/L2 routine fans out to threads
 * (L1 family default; see the family-override note above). */
#ifndef MULTI_THREAD_MINIMAL
#define MULTI_THREAD_MINIMAL 10000
#endif

/* Upper bound on the partition count used by the partitioned packed /
 * symmetric L2 updates. */
#ifndef MAX_PARTITION_CPUS
#define MAX_PARTITION_CPUS   256
#endif

/* Adaptive-MC L2 cache target for the L3 drivers (P-core L2 nominal):
 * when K fits in one KC panel, MC is grown so MC*KC roughly fills this. */
#define L2_TARGET_BYTES (256L * 1024L)

/* Per-thread slot cap for the L1 reductions' partial[] / pidx[] / pmax[]
 * stack arrays (max OpenMP team size those paths will use). */
#define L1_PARTIAL_MAX_THREADS 64

#endif /* EPBLAS_OPENBLAS_KIND10_EBLAS_TUNING_H */
