//
// Created by 赵丹 on 2023/5/7.
//

#ifndef GEMM_OPT_MATMUL_1X4_6_H
#define GEMM_OPT_MATMUL_1X4_6_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

// In this version, we accumulate in registers and put A( 0, p ) in a register
void AddDot1x4_6(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    register float c_00_reg, c_01_reg, c_02_reg, c_03_reg; // hold C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    register float a_0p_reg;  // holds A( 0, p )

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;

    for (int p = 0; p < k; p++) {
        a_0p_reg = A(0, p);
        c_00_reg += a_0p_reg * B(p, 0);
        c_01_reg += a_0p_reg * B(p, 1);
        c_02_reg += a_0p_reg * B(p, 2);
        c_03_reg += a_0p_reg * B(p, 3);
    }
    C(0, 0) += c_00_reg;
    C(0, 1) += c_01_reg;
    C(0, 2) += c_02_reg;
    C(0, 3) += c_03_reg;
}

void my_matmul_1x4_6(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) {
            AddDot1x4_6(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

#endif //GEMM_OPT_MATMUL_1X4_6_H
