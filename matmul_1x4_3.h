//
// Created by richard on 5/6/23.
//

#ifndef GEMM_OPT_MATMUL_1X4_3_H
#define GEMM_OPT_MATMUL_1X4_3_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

// Create macro to let Y(i) equal the ith element of x
#define Y(i) y[(i) * incx]

// compute gamma := x' * y + gamma with vectors x and y of length n.
// Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
void AddDot(int k, const float *x, int incx, const float *y, float *gamma) {
    for (int p = 0; p < k; p++) {
        *gamma += x[p] * Y(p);
    }
}

void AddDot1x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
}

void my_matmul_1x4_3(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) {
            AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}


#endif //GEMM_OPT_MATMUL_1X4_3_H
