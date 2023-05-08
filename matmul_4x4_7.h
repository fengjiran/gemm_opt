//
// Created by 赵丹 on 2023/5/8.
//

#ifndef GEMM_OPT_MATMUL_4X4_7_H
#define GEMM_OPT_MATMUL_4X4_7_H

#define A(i, j) a[ (i) * lda + (j) ]
#define B(i, j) b[ (i) * ldb + (j) ]
#define C(i, j) c[ (i) * ldc + (j) ]

void AddDot4x4_7(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    register float c_00_reg, c_01_reg, c_02_reg, c_03_reg,
            c_10_reg, c_11_reg, c_12_reg, c_13_reg,
            c_20_reg, c_21_reg, c_22_reg, c_23_reg,
            c_30_reg, c_31_reg, c_32_reg, c_33_reg,
            a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;
    c_10_reg = 0;
    c_11_reg = 0;
    c_12_reg = 0;
    c_13_reg = 0;
    c_20_reg = 0;
    c_21_reg = 0;
    c_22_reg = 0;
    c_23_reg = 0;
    c_30_reg = 0;
    c_31_reg = 0;
    c_32_reg = 0;
    c_33_reg = 0;

    const float *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;

    for (int p = 0; p < k; p++) {
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        b_p0_ptr = &B(p, 0);
        b_p1_ptr = &B(p, 1);
        b_p2_ptr = &B(p, 2);
        b_p3_ptr = &B(p, 3);

        // first row
        c_00_reg += a_0p_reg * *b_p0_ptr;
        c_01_reg += a_0p_reg * *b_p1_ptr;
        c_02_reg += a_0p_reg * *b_p2_ptr;
        c_03_reg += a_0p_reg * *b_p3_ptr;

        // second row
        c_10_reg += a_1p_reg * *b_p0_ptr;
        c_11_reg += a_1p_reg * *b_p1_ptr;
        c_12_reg += a_1p_reg * *b_p2_ptr;
        c_13_reg += a_1p_reg * *b_p3_ptr;

        // third row
        c_20_reg += a_2p_reg * *b_p0_ptr;
        c_21_reg += a_2p_reg * *b_p1_ptr;
        c_22_reg += a_2p_reg * *b_p2_ptr;
        c_23_reg += a_2p_reg * *b_p3_ptr;

        // forth row
        c_30_reg += a_3p_reg * *b_p0_ptr;
        c_31_reg += a_3p_reg * *b_p1_ptr;
        c_32_reg += a_3p_reg * *b_p2_ptr;
        c_33_reg += a_3p_reg * *b_p3_ptr;
    }

    C(0, 0) += c_00_reg;
    C(0, 1) += c_01_reg;
    C(0, 2) += c_02_reg;
    C(0, 3) += c_03_reg;
    C(1, 0) += c_10_reg;
    C(1, 1) += c_11_reg;
    C(1, 2) += c_12_reg;
    C(1, 3) += c_13_reg;
    C(2, 0) += c_20_reg;
    C(2, 1) += c_21_reg;
    C(2, 2) += c_22_reg;
    C(2, 3) += c_23_reg;
    C(3, 0) += c_30_reg;
    C(3, 1) += c_31_reg;
    C(3, 2) += c_32_reg;
    C(3, 3) += c_33_reg;
}

void my_matmul_4x4_7(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            AddDot4x4_7(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

#endif //GEMM_OPT_MATMUL_4X4_7_H
