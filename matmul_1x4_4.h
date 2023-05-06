//
// Created by 赵丹 on 2023/5/7.
//

#ifndef GEMM_OPT_MATMUL_1X4_4_H
#define GEMM_OPT_MATMUL_1X4_4_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

// In this version, we "inline" AddDot
void AddDot1x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    //
}

#endif //GEMM_OPT_MATMUL_1X4_4_H
