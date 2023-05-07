//
// Created by richard on 5/7/23.
//

#ifndef GEMM_OPT_MATMUL_4X4_4_H
#define GEMM_OPT_MATMUL_4X4_4_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

#define Y(i) y[(i) * incx]

void AddDot4x4_4(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    // first row
    for (int p = 0; p < k; p++) {
        C(0, 0) += A(0, p) * B(p, 0);
    }

    for (int p = 0; p < k; p++) {
        C(0, 1) += A(0, p) * B(p, 1);
    }

    for (int p = 0; p < k; p++) {
        C(0, 2) += A(0, p) * B(p, 2);
    }

    for (int p = 0; p < k; p++) {
        C(0, 3) += A(0, p) * B(p, 3);
    }

    // second row
    for (int p = 0; p < k; p++) {
        C(1, 0) += A(1, p) * B(p, 0);
    }

    for (int p = 0; p < k; p++) {
        C(1, 1) += A(1, p) * B(p, 1);
    }

    for (int p = 0; p < k; p++) {
        C(1, 2) += A(1, p) * B(p, 2);
    }

    for (int p = 0; p < k; p++) {
        C(1, 3) += A(1, p) * B(p, 3);
    }

    // third row
    for (int p = 0; p < k; p++) {
        C(2, 0) += A(2, p) * B(p, 0);
    }

    for (int p = 0; p < k; p++) {
        C(2, 1) += A(2, p) * B(p, 1);
    }

    for (int p = 0; p < k; p++) {
        C(2, 2) += A(2, p) * B(p, 2);
    }

    for (int p = 0; p < k; p++) {
        C(2, 3) += A(2, p) * B(p, 3);
    }

    // forth row
    for (int p = 0; p < k; p++) {
        C(3, 0) += A(3, p) * B(p, 0);
    }

    for (int p = 0; p < k; p++) {
        C(3, 1) += A(3, p) * B(p, 1);
    }

    for (int p = 0; p < k; p++) {
        C(3, 2) += A(3, p) * B(p, 2);
    }

    for (int p = 0; p < k; p++) {
        C(3, 3) += A(3, p) * B(p, 3);
    }
}

void my_matmul_4x4_4(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            AddDot4x4_4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

#endif //GEMM_OPT_MATMUL_4X4_4_H
