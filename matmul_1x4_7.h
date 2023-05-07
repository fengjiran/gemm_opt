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
    register float b_0p_reg;  // holds B(p, 0)

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;
}

#endif //GEMM_OPT_MATMUL_1X4_7_H
