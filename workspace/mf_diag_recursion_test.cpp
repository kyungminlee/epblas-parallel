/*
 * Standalone before/after driver for the multifloats diag_dispatch
 * self-recursion bug. Each rank-k/2k routine's diag block falls back to a
 * scalar path when K > kMaxK (512) or jb > kMaxBlockM. The buggy fallback
 * self-recurses -> infinite recursion -> stack overflow (SIGSEGV).
 *
 * Trigger: K = 600 (> 512) with small N (jb = N <= 128) so the diagonal
 * block takes the scalar path. Oracle: the same product split into two
 * K = 300 calls (both <= 512 -> known-good SIMD path); results must match.
 *
 * BEFORE fix: SIGSEGV on the first routine.
 * AFTER  fix: all routines print PASS.
 */
#include <multifloats.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

static R rdd(double v) { return R{v, 0.0}; }
static T cx(double re, double im) { return T{R{re, 0.0}, R{im, 0.0}}; }

extern "C" {
void wsyrk_(const char*, const char*, const int*, const int*, const T*,
            const T*, const int*, const T*, T*, const int*, std::size_t, std::size_t);
void wherk_(const char*, const char*, const int*, const int*, const R*,
            const T*, const int*, const R*, T*, const int*, std::size_t, std::size_t);
void wsyr2k_(const char*, const char*, const int*, const int*, const T*,
             const T*, const int*, const T*, const int*, const T*, T*, const int*,
             std::size_t, std::size_t);
void wher2k_(const char*, const char*, const int*, const int*, const T*,
             const T*, const int*, const T*, const int*, const R*, T*, const int*,
             std::size_t, std::size_t);
void msyr2k_(const char*, const char*, const int*, const int*, const R*,
             const R*, const int*, const R*, const int*, const R*, R*, const int*,
             std::size_t, std::size_t);
}

static const int N = 8;
static const int KBIG = 600, KHALF = 300;
static const char L = 'L', NT = 'N';

// deterministic pseudo-random data
static double dv(int i, int j, int salt) {
    return 0.5 * std::sin(0.7 * i + 1.3 * j + 0.31 * salt);
}

static int fails = 0;

static void check(const char* name, const double* ref, const double* test, int n2) {
    double maxrel = 0.0;
    for (int i = 0; i < n2; ++i) {
        double a = ref[i], b = test[i];
        double d = std::fabs(a - b);
        double s = std::fabs(a) + std::fabs(b);
        double rel = (s > 1e-300) ? d / s : d;
        if (rel > maxrel) maxrel = rel;
    }
    bool ok = maxrel < 1e-12;
    std::printf("%-8s K=%d split-vs-full maxrel=%.3e  %s\n",
                name, KBIG, maxrel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
}

// extract hi limbs of lower triangle into a flat double array (re,im pairs)
static void dumpC_cx(const T* C, double* out) {
    int p = 0;
    for (int j = 0; j < N; ++j)
        for (int i = j; i < N; ++i) { out[p++] = C[(size_t)j*N+i].re.limbs[0];
                                      out[p++] = C[(size_t)j*N+i].im.limbs[0]; }
}
static void dumpC_r(const R* C, double* out) {
    int p = 0;
    for (int j = 0; j < N; ++j)
        for (int i = j; i < N; ++i) out[p++] = C[(size_t)j*N+i].limbs[0];
}

int main() {
    const T zc = cx(0,0), oc = cx(1,0), alpha_c = cx(1.5, -0.4);
    const R zr = rdd(0), orr = rdd(1), alpha_r = rdd(1.7), beta_r = rdd(0.5);
    const T beta_c = cx(0.5, 0.0);

    // ---- complex A,B (N x KBIG col-major, lda=N) ----
    std::vector<T> A((size_t)N*KBIG), B((size_t)N*KBIG), C0((size_t)N*N);
    for (int j = 0; j < KBIG; ++j) for (int i = 0; i < N; ++i) {
        A[(size_t)j*N+i] = cx(dv(i,j,0), dv(i,j,1));
        B[(size_t)j*N+i] = cx(dv(i,j,2), dv(i,j,3));
    }
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
        C0[(size_t)j*N+i] = cx(dv(i,j,4), dv(i,j,5));

    double ref[2*N*N], test[2*N*N];

    // ===== wsyrk (TR='N'): C = beta*C + alpha*A*A^T =====
    {
        std::vector<T> Cr = C0, Ct = C0;
        wsyrk_(&L,&NT,&N,&KHALF,&alpha_c, A.data(),&N, &beta_c, Cr.data(),&N,1,1);
        const T* A2 = A.data() + (size_t)KHALF*N;
        wsyrk_(&L,&NT,&N,&KHALF,&alpha_c, A2,&N, &oc, Cr.data(),&N,1,1);
        wsyrk_(&L,&NT,&N,&KBIG,&alpha_c, A.data(),&N, &beta_c, Ct.data(),&N,1,1);
        dumpC_cx(Cr.data(), ref); dumpC_cx(Ct.data(), test);
        check("wsyrk", ref, test, N*(N+1));   // (N*(N+1)/2)*2
    }
    // ===== wherk (TR='N', real alpha/beta): C = beta*C + alpha*A*A^H =====
    {
        std::vector<T> Cr = C0, Ct = C0;
        wherk_(&L,&NT,&N,&KHALF,&alpha_r, A.data(),&N, &beta_r, Cr.data(),&N,1,1);
        const T* A2 = A.data() + (size_t)KHALF*N;
        wherk_(&L,&NT,&N,&KHALF,&alpha_r, A2,&N, &orr, Cr.data(),&N,1,1);
        wherk_(&L,&NT,&N,&KBIG,&alpha_r, A.data(),&N, &beta_r, Ct.data(),&N,1,1);
        dumpC_cx(Cr.data(), ref); dumpC_cx(Ct.data(), test);
        check("wherk", ref, test, N*(N+1));
    }
    // ===== wsyr2k (TR='N'): C = alpha*(A*B^T + B*A^T) + beta*C =====
    {
        std::vector<T> Cr = C0, Ct = C0;
        const T* A2 = A.data() + (size_t)KHALF*N;
        const T* B2 = B.data() + (size_t)KHALF*N;
        wsyr2k_(&L,&NT,&N,&KHALF,&alpha_c, A.data(),&N, B.data(),&N, &beta_c, Cr.data(),&N,1,1);
        wsyr2k_(&L,&NT,&N,&KHALF,&alpha_c, A2,&N, B2,&N, &oc, Cr.data(),&N,1,1);
        wsyr2k_(&L,&NT,&N,&KBIG,&alpha_c, A.data(),&N, B.data(),&N, &beta_c, Ct.data(),&N,1,1);
        dumpC_cx(Cr.data(), ref); dumpC_cx(Ct.data(), test);
        check("wsyr2k", ref, test, N*(N+1));
    }
    // ===== wher2k (TR='N', real beta): C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C =====
    {
        std::vector<T> Cr = C0, Ct = C0;
        const T* A2 = A.data() + (size_t)KHALF*N;
        const T* B2 = B.data() + (size_t)KHALF*N;
        wher2k_(&L,&NT,&N,&KHALF,&alpha_c, A.data(),&N, B.data(),&N, &beta_r, Cr.data(),&N,1,1);
        wher2k_(&L,&NT,&N,&KHALF,&alpha_c, A2,&N, B2,&N, &orr, Cr.data(),&N,1,1);
        wher2k_(&L,&NT,&N,&KBIG,&alpha_c, A.data(),&N, B.data(),&N, &beta_r, Ct.data(),&N,1,1);
        dumpC_cx(Cr.data(), ref); dumpC_cx(Ct.data(), test);
        check("wher2k", ref, test, N*(N+1));
    }
    // ===== msyr2k (REAL DD, TR='N'): C = alpha*(A*B^T + B*A^T) + beta*C =====
    {
        std::vector<R> Ar((size_t)N*KBIG), Br((size_t)N*KBIG), Cr0((size_t)N*N);
        for (int j = 0; j < KBIG; ++j) for (int i = 0; i < N; ++i) {
            Ar[(size_t)j*N+i] = rdd(dv(i,j,0));
            Br[(size_t)j*N+i] = rdd(dv(i,j,2));
        }
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
            Cr0[(size_t)j*N+i] = rdd(dv(i,j,4));
        std::vector<R> Cr = Cr0, Ct = Cr0;
        const R* A2 = Ar.data() + (size_t)KHALF*N;
        const R* B2 = Br.data() + (size_t)KHALF*N;
        msyr2k_(&L,&NT,&N,&KHALF,&alpha_r, Ar.data(),&N, Br.data(),&N, &beta_r, Cr.data(),&N,1,1);
        msyr2k_(&L,&NT,&N,&KHALF,&alpha_r, A2,&N, B2,&N, &orr, Cr.data(),&N,1,1);
        msyr2k_(&L,&NT,&N,&KBIG,&alpha_r, Ar.data(),&N, Br.data(),&N, &beta_r, Ct.data(),&N,1,1);
        dumpC_r(Cr.data(), ref); dumpC_r(Ct.data(), test);
        check("msyr2k", ref, test, N*(N+1)/2);
    }

    std::printf(fails ? "\n%d FAILURES\n" : "\nALL PASS\n", fails);
    return fails ? 1 : 0;
}
