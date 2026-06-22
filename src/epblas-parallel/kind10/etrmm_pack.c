/*
 * etrmm_pack — kind10 (REAL(KIND=10) / long double) TRMM A-side triangular
 * packers. Faithful ports of OpenBLAS kernel/generic/trmm_{ut,un,lt,ln}copy_2.c
 * with the compile-time UNIT macro replaced by a runtime `unit` flag
 * (`ONE`/`ZERO` → `1.0L`/`0.0L`). Same code the openblas overlay carries as
 * eblas_etrmm_i{ut,un,lt,ln}copy in eblas_l3_real.c.
 *
 * Each packs the relevant triangle of A into the packed buffer in the
 * ob contiguous-odd-tail convention (consumed by etrmm_kernel / etri_kernel).
 * `posX`/`posY` position the diagonal relative to the packer's local 2-step
 * sweep; `unit` selects unit-diagonal (writes 1.0L on the diagonal).
 *
 * The same four packers serve both SIDE=L (TRMM_I*COPY role) and SIDE=R
 * (TRMM_O*COPY role) — same function, different call-site context; the
 * (uplo,trans)→packer mapping differs by SIDE (see pack_trmm_a in
 * etrmm_serial.c).
 */

#include <stddef.h>

#include "etrmm_kernel.h"

typedef etrmm_T T;

void etrmm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, bool unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            } else {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X < posY) {
                        ao1 += 2;
                        ao2 += 2;
                        b   += 4;
                    } else if (X > posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data02;
                        b[2] = data03;
                        b[3] = data04;

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data03 = *(ao2 + 0);
                            b[0] = 1.0L;
                            b[1] = 0.0L;
                            b[2] = data03;
                            b[3] = 1.0L;
                        } else {
                            data01 = *(ao1 + 0);
                            data03 = *(ao2 + 0);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = 0.0L;
                            b[2] = data03;
                            b[3] = data04;
                        }

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X < posY) {
                    ao1 += 1;
                    ao2 += 1;
                    b += 2;
                } else if (X > posY) {
                    data01 = *(ao1 + 0);
                    data02 = *(ao1 + 1);
                    b[0] = data01;
                    b[1] = data02;
                    ao1 += lda;
                    b += 2;
                } else {
                    if (unit) {
                        b[0] = 1.0L;
                        b[1] = 0.0L;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                        b[1] = 0.0L;
                    }
                    ao1 += lda;
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posX + (posY + 0) * lda;
        } else {
            ao1 = a + posY + (posX + 0) * lda;
        }

        i = m;
        if (m > 0) {
            do {
                if (X < posY) {
                    b += 1;
                    ao1 += 1;
                } else if (X > posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    b += 1;
                    ao1 += lda;
                } else {
                    if (unit) {
                        b[0] = 1.0L;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += lda;
                }

                X += 1;
                i--;
            } while (i > 0);
        }
    }
}


void etrmm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, bool unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            } else {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X < posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data03;
                        b[2] = data02;
                        b[3] = data04;

                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    } else if (X > posY) {
                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data03 = *(ao2 + 0);
                            b[0] = 1.0L;
                            b[1] = data03;
                            b[2] = 0.0L;
                            b[3] = 1.0L;
                        } else {
                            data01 = *(ao1 + 0);
                            data03 = *(ao2 + 0);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = data03;
                            b[2] = 0.0L;
                            b[3] = data04;
                        }

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X < posY) {
                    data01 = *(ao1 + 0);
                    data03 = *(ao2 + 0);
                    b[0] = data01;
                    b[1] = data03;
                    b += 2;
                } else if (X > posY) {
                    b += 2;
                } else {
                    if (unit) {
                        data03 = *(ao2 + 0);
                        b[0] = 1.0L;
                        b[1] = data03;
                    } else {
                        data01 = *(ao1 + 0);
                        data03 = *(ao2 + 0);
                        b[0] = data01;
                        b[1] = data03;
                    }
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posX + (posY + 0) * lda;
        } else {
            ao1 = a + posY + (posX + 0) * lda;
        }

        i = m;
        if (m > 0) {
            do {
                if (X < posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    ao1 += 1;
                    b += 1;
                } else if (X > posY) {
                    ao1 += lda;
                    b += 1;
                } else {
                    if (unit) {
                        b[0] = 1.0L;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += lda;
                }

                X += 1;
                i--;
            } while (i > 0);
        }
    }
}


void etrmm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, bool unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            } else {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X > posY) {
                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    } else if (X < posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data02;
                        b[2] = data03;
                        b[3] = data04;

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data02 = *(ao1 + 1);
                            b[0] = 1.0L;
                            b[1] = data02;
                            b[2] = 0.0L;
                            b[3] = 1.0L;
                        } else {
                            data01 = *(ao1 + 0);
                            data02 = *(ao1 + 1);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = data02;
                            b[2] = 0.0L;
                            b[3] = data04;
                        }

                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X > posY) {
                    ao1 += 1;
                    ao2 += 1;
                    b += 2;
                } else if (X < posY) {
                    data01 = *(ao1 + 0);
                    data02 = *(ao1 + 1);
                    b[0] = data01;
                    b[1] = data02;
                    ao1 += lda;
                    b += 2;
                } else {
                    if (unit) {
                        data02 = *(ao1 + 1);
                        b[0] = 1.0L;
                        b[1] = data02;
                    } else {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        b[0] = data01;
                        b[1] = data02;
                    }
                    ao1 += 2;
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posY + (posX + 0) * lda;
        } else {
            ao1 = a + posX + (posY + 0) * lda;
        }

        i = m;
        if (i > 0) {
            do {
                if (X > posY) {
                    ao1 += 1;
                    b += 1;
                } else if (X < posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    ao1 += lda;
                    b += 1;
                } else {
                    if (unit) {
                        b[0] = 1.0L;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += 1;
                }

                X++;
                i--;
            } while (i > 0);
        }
    }
}


void etrmm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, bool unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            } else {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X > posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data03;
                        b[2] = data02;
                        b[3] = data04;

                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    } else if (X < posY) {
                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data02 = *(ao1 + 1);
                            b[0] = 1.0L;
                            b[1] = 0.0L;
                            b[2] = data02;
                            b[3] = 1.0L;
                        } else {
                            data01 = *(ao1 + 0);
                            data02 = *(ao1 + 1);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = 0.0L;
                            b[2] = data02;
                            b[3] = data04;
                        }
                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X > posY) {
                    data01 = *(ao1 + 0);
                    data03 = *(ao2 + 0);
                    b[0] = data01;
                    b[1] = data03;
                    b += 2;
                } else if (X < posY) {
                    b += 2;
                } else {
                    if (unit) {
                        data03 = *(ao2 + 0);
                        b[0] = 1.0L;
                        b[1] = data03;
                    } else {
                        data01 = *(ao1 + 0);
                        data03 = *(ao2 + 0);
                        b[0] = data01;
                        b[1] = data03;
                    }
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posY + (posX + 0) * lda;
        } else {
            ao1 = a + posX + (posY + 0) * lda;
        }

        i = m;
        if (i > 0) {
            do {
                if (X > posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    ao1 += 1;
                    b += 1;
                } else if (X < posY) {
                    ao1 += lda;
                    b += 1;
                } else {
                    if (unit) {
                        b[0] = 1.0L;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += 1;
                }

                X++;
                i--;
            } while (i > 0);
        }
    }
}

