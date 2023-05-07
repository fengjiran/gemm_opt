//
// Created by 赵丹 on 2023/5/7.
//

#ifndef GEMM_OPT_MATMUL_1X4_5_H
#define GEMM_OPT_MATMUL_1X4_5_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

// In this version, we merge the four loops, computing four inner
// products simultaneously.
void AddDot1x4_5(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    // AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    // AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    // AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    // AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
    for (int p = 0; p < k; p++) {
        C(0, 0) += A(0, p) * B(p, 0);
        C(0, 1) += A(0, p) * B(p, 1);
        C(0, 2) += A(0, p) * B(p, 2);
        C(0, 3) += A(0, p) * B(p, 3);
    }
}

void my_matmul_1x4_5(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) {
            AddDot1x4_5(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

#endif //GEMM_OPT_MATMUL_1X4_5_H
