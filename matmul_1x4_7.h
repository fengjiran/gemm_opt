//
// Created by 赵丹 on 2023/5/7.
//

#ifndef GEMM_OPT_MATMUL_1X4_7_H
#define GEMM_OPT_MATMUL_1X4_7_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

// In this version, we use pointer to track where in four rows of A we are
void AddDot1x4_7(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    register float c_00_reg, c_01_reg, c_02_reg, c_03_reg; // hold C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    register float b_p0_reg;  // holds B(p, 0)

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;

    const float *ap0_ptr = &A(0, 0);
    const float *ap1_ptr = &A(1, 0);
    const float *ap2_ptr = &A(2, 0);
    const float *ap3_ptr = &A(3, 0);

    for (int p = 0; p < k; p++) {
        b_p0_reg = B(p, 0);
        c_00_reg += b_p0_reg * (*ap0_ptr++);
        c_01_reg += b_p0_reg * (*ap1_ptr++);
        c_02_reg += b_p0_reg * (*ap2_ptr++);
        c_03_reg += b_p0_reg * (*ap3_ptr++);
    }

    C(0, 0) += c_00_reg;
    C(1, 0) += c_01_reg;
    C(2, 0) += c_02_reg;
    C(3, 0) += c_03_reg;
}

void my_matmul_1x4_7(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i += 4) {
            AddDot1x4_7(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

#endif //GEMM_OPT_MATMUL_1X4_7_H
