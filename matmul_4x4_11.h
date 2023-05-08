//
// Created by 赵丹 on 2023/5/8.
//

#ifndef GEMM_OPT_MATMUL_4X4_11_H
#define GEMM_OPT_MATMUL_4X4_11_H

#include <mmintrin.h>  // MMX
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i) : (j))

// block size
#define MC 256
#define KC 128

typedef union {
    __m128 v;
    float d[4];
} v2df_t;

void AddDot4x4_11(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    const float *a_0p_ptr = &A(0, 0);
    const float *a_1p_ptr = &A(1, 0);
    const float *a_2p_ptr = &A(2, 0);
    const float *a_3p_ptr = &A(3, 0);

    v2df_t c_p0_sum, c_p1_sum, c_p2_sum, c_p3_sum;
    v2df_t a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;
    v2df_t b_reg;

    c_p0_sum.v = _mm_setzero_ps();
    c_p1_sum.v = _mm_setzero_ps();
    c_p2_sum.v = _mm_setzero_ps();
    c_p3_sum.v = _mm_setzero_ps();

    a_0p_reg.v = _mm_setzero_ps();
    a_1p_reg.v = _mm_setzero_ps();
    a_2p_reg.v = _mm_setzero_ps();
    a_3p_reg.v = _mm_setzero_ps();

    for (int p = 0; p < k; p++) {
        b_reg.v = _mm_load_ps((float *) &B(p, 0));
        a_0p_reg.v = _mm_set_ps1(*a_0p_ptr++);
        a_1p_reg.v = _mm_set_ps1(*a_1p_ptr++);
        a_2p_reg.v = _mm_set_ps1(*a_2p_ptr++);
        a_3p_reg.v = _mm_set_ps1(*a_3p_ptr++);

        c_p0_sum.v += b_reg.v * a_0p_reg.v;
        c_p1_sum.v += b_reg.v * a_1p_reg.v;
        c_p2_sum.v += b_reg.v * a_2p_reg.v;
        c_p3_sum.v += b_reg.v * a_3p_reg.v;
    }

    C(0, 0) += c_p0_sum.d[0];
    C(0, 1) += c_p0_sum.d[1];
    C(0, 2) += c_p0_sum.d[2];
    C(0, 3) += c_p0_sum.d[3];

    C(1, 0) += c_p1_sum.d[0];
    C(1, 1) += c_p1_sum.d[1];
    C(1, 2) += c_p1_sum.d[2];
    C(1, 3) += c_p1_sum.d[3];

    C(2, 0) += c_p2_sum.d[0];
    C(2, 1) += c_p2_sum.d[1];
    C(2, 2) += c_p2_sum.d[2];
    C(2, 3) += c_p2_sum.d[3];

    C(3, 0) += c_p3_sum.d[0];
    C(3, 1) += c_p3_sum.d[1];
    C(3, 2) += c_p3_sum.d[2];
    C(3, 3) += c_p3_sum.d[3];
}

void InnerKernel(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            AddDot4x4_11(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void my_matmul_4x4_11(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    int pb, ib;
    for (int p = 0; p < k; p += KC) {
        pb = min(k - p, KC);
        for (int i = 0; i < m; i += MC) {
            ib = min(m - i, MC);
            InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
        }
    }
}

#endif //GEMM_OPT_MATMUL_4X4_11_H
