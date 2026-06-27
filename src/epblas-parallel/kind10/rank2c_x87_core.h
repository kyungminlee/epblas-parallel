/*
 * rank2c_x87_core.h — shared x87 inline-asm core for kind10 complex rank-2
 * updates (A += x·t1 + y·t2 per element, t1/t2 complex loop invariants).
 *
 * The off-diagonal run c[i] += xc[i]·t1 + yc[i]·t2 is identical for the packed
 * (yhpr2) and full-storage (yher2) Hermitian rank-2 updates — only the column
 * base offset and the diagonal write differ, and those stay in C. This is the
 * hand-written transcription of gfortran's strided `.L40` loop, the fastest code
 * any compiler emits for this fp80 kernel (equal 9 fldt/element to gcc's, but a
 * better x87 schedule gcc will not reproduce from C; see task 16 /
 * project_l2_strided_gather). The 4 loop-invariant constants stay resident on
 * the x87 stack (top->bottom t1i,t2r,t1r,t2i); pointer increments interleaved as
 * gfortran places them (c early with -16/-32 back-offsets, x mid, y late);
 * left-folded ((c + x·t1) + y·t2) association — bit-exact (fuzz relerr 0).
 *
 * yr2c_run        — unit-stride x/y (native-contiguous or gathered scratch).
 * yr2c_run_strided — x/y advance by caller byte strides sx/sy (incx*sizeof(TC),
 *                    incy*sizeof(TC)); c stays contiguous (+32). Lets the serial
 *                    strided path walk inputs in place, no gather.
 *
 * no_stack_protector: the "m" constant operands + "memory" clobber otherwise
 * trip a per-column canary prologue gfortran's inlined loop lacks; the asm
 * touches no stack buffer. Complex params aliased as {re,im} pairs so constants
 * load straight from their incoming slots (4 fp80 mem ops/column, not 12).
 */
#ifndef EPBLAS_KIND10_RANK2C_X87_CORE_H
#define EPBLAS_KIND10_RANK2C_X87_CORE_H

#include <stddef.h>

typedef _Complex long double yr2c_TC;
typedef long double yr2c_TR;

__attribute__((always_inline, no_stack_protector))
static inline void yr2c_run(ptrdiff_t mo, yr2c_TC t1, yr2c_TC t2,
                      const yr2c_TC *restrict xc, const yr2c_TC *restrict yc,
                      yr2c_TC *restrict c) {
    if (mo <= 0) return;
    const yr2c_TR *p1 = (const yr2c_TR *)&t1, *p2 = (const yr2c_TR *)&t2;
    const yr2c_TC *end = c + mo;
    __asm__ volatile(
        "fldt %[t2i]\n\t"
        "fldt %[t1r]\n\t"
        "fldt %[t2r]\n\t"
        "fldt %[t1i]\n\t"
        ".p2align 4\n\t"
        ".p2align 3\n\t"
        "1:\n\t"
        "fldt (%[x])\n\t"
        "addq $32, %[c]\n\t"
        "fldt 16(%[x])\n\t"
        "fmul %%st(4), %%st\n\t"
        "fld %%st(2)\n\t"
        "fmul %%st(2), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt -16(%[c])\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"
        "fmul %%st(6), %%st\n\t"
        "fldt 16(%[y])\n\t"
        "fmul %%st(5), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fxch %%st(1)\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[x])\n\t"
        "addq $32, %[x]\n\t"
        "fmul %%st(3), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"
        "fldt -32(%[c])\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[y])\n\t"
        "addq $32, %[y]\n\t"
        "fmul %%st(7), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fstpt -32(%[c])\n\t"
        "fstpt -16(%[c])\n\t"
        "cmpq %[end], %[c]\n\t"
        "jne 1b\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        : [x] "+r"(xc), [y] "+r"(yc), [c] "+r"(c)
        : [end] "r"(end),
          [t1r] "m"(p1[0]), [t1i] "m"(p1[1]), [t2r] "m"(p2[0]), [t2i] "m"(p2[1])
        : "memory", "cc",
          "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)");
}

__attribute__((always_inline, no_stack_protector))
static inline void yr2c_run_strided(ptrdiff_t mo, yr2c_TC t1, yr2c_TC t2,
                      const yr2c_TC *restrict xc, const yr2c_TC *restrict yc,
                      yr2c_TC *restrict c, ptrdiff_t sx, ptrdiff_t sy) {
    if (mo <= 0) return;
    const yr2c_TR *p1 = (const yr2c_TR *)&t1, *p2 = (const yr2c_TR *)&t2;
    const yr2c_TC *end = c + mo;
    __asm__ volatile(
        "fldt %[t2i]\n\t"
        "fldt %[t1r]\n\t"
        "fldt %[t2r]\n\t"
        "fldt %[t1i]\n\t"
        ".p2align 4\n\t"
        ".p2align 3\n\t"
        "1:\n\t"
        "fldt (%[x])\n\t"
        "addq $32, %[c]\n\t"
        "fldt 16(%[x])\n\t"
        "fmul %%st(4), %%st\n\t"
        "fld %%st(2)\n\t"
        "fmul %%st(2), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt -16(%[c])\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"
        "fmul %%st(6), %%st\n\t"
        "fldt 16(%[y])\n\t"
        "fmul %%st(5), %%st\n\t"
        "faddp %%st, %%st(1)\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fxch %%st(1)\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[x])\n\t"
        "addq %[sx], %[x]\n\t"      /* X += incx*sizeof(TC) */
        "fmul %%st(3), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"
        "fldt -32(%[c])\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fldt (%[y])\n\t"
        "fmul %%st(4), %%st\n\t"
        "fldt 16(%[y])\n\t"
        "addq %[sy], %[y]\n\t"      /* Y += incy*sizeof(TC) */
        "fmul %%st(7), %%st\n\t"
        "fsubrp %%st, %%st(1)\n\t"
        "faddp %%st, %%st(1)\n\t"
        "fstpt -32(%[c])\n\t"
        "fstpt -16(%[c])\n\t"
        "cmpq %[end], %[c]\n\t"
        "jne 1b\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        "fstp %%st(0)\n\t"
        : [x] "+r"(xc), [y] "+r"(yc), [c] "+r"(c)
        : [end] "r"(end), [sx] "r"(sx), [sy] "r"(sy),
          [t1r] "m"(p1[0]), [t1i] "m"(p1[1]), [t2r] "m"(p2[0]), [t2i] "m"(p2[1])
        : "memory", "cc",
          "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)");
}

#endif /* EPBLAS_KIND10_RANK2C_X87_CORE_H */
