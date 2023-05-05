//
// Created by richard on 5/6/23.
//

#ifndef GEMM_OPT_MATMUL_ORIGIN_H
#define GEMM_OPT_MATMUL_ORIGIN_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

// gemm C = A * B + C
void matmul_origin(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}

#endif //GEMM_OPT_MATMUL_ORIGIN_H
