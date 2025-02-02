//
// Created by richard on 5/8/23.
//
#include "utils.h"

void random_matrix(int m, int n, float* a, int lda) {
    //double drand48();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A(i, j) = static_cast<float>(drand48());
        }
    }
}

void copy_matrix(int m, int n, const float* a, int lda, float* b, int ldb) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B(i, j) = A(i, j);
        }
    }
}

float compare_matrices(int m, int n, float* a, int lda, float* b, int ldb) {
    float max_diff = 0.0, diff;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            diff = std::abs(A(i, j) - B(i, j));
            max_diff = max(diff, max_diff);
            if (max_diff > 0.5f || max_diff < -0.5f) {
                printf("\n error: i %d  j %d diff %f", i, j, max_diff);
            }
        }
    }
    return max_diff;
}

// gemm C = A * B + C
void matmul_origin(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C(i, j) = C(i, j) + A(i, p) * B(p, j);
            }
        }
    }
}

// compute gamma := x' * y + gamma with vectors x and y of length n.
// Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
void AddDot(int k, const float* x, int incx, const float* y, float* gamma) {
    for (int p = 0; p < k; p++) {
        *gamma += x[p] * Y(p);
    }
}

void AddDot1x4_3(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
}

void my_matmul_1x4_3(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) {
            AddDot1x4_3(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

// In this version, we "inline" AddDot
void AddDot1x4_4(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    // AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    for (int p = 0; p < k; p++) {
        C(0, 0) += A(0, p) * B(p, 0);
    }

    // AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    for (int p = 0; p < k; p++) {
        C(0, 1) += A(0, p) * B(p, 1);
    }

    // AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    for (int p = 0; p < k; p++) {
        C(0, 2) += A(0, p) * B(p, 2);
    }

    // AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
    for (int p = 0; p < k; p++) {
        C(0, 3) += A(0, p) * B(p, 3);
    }
}

void my_matmul_1x4_4(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) {
            AddDot1x4_4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

// In this version, we merge the four loops, computing four inner
// products simultaneously.
void AddDot1x4_5(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
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

void my_matmul_1x4_5(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) { AddDot1x4_5(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

// In this version, we accumulate in registers and put A( 0, p ) in a register
void AddDot1x4_6(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg;// hold C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    float a_0p_reg;                              // holds A( 0, p )

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

void my_matmul_1x4_6(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) { AddDot1x4_6(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

// In this version, we use pointer to track where in four rows of A we are
void AddDot1x4_7(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg;// hold C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    float b_p0_reg;                              // holds B(p, 0)

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;

    const float* ap0_ptr = &A(0, 0);
    const float* ap1_ptr = &A(1, 0);
    const float* ap2_ptr = &A(2, 0);
    const float* ap3_ptr = &A(3, 0);

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

void my_matmul_1x4_7(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i += 4) { AddDot1x4_7(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

// loop unrolling
void AddDot1x4_8(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg;// hold C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    float b_p0_reg;                              // holds B(p, 0)

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;

    const float* ap0_ptr = &A(0, 0);
    const float* ap1_ptr = &A(1, 0);
    const float* ap2_ptr = &A(2, 0);
    const float* ap3_ptr = &A(3, 0);

    for (int p = 0; p < k; p += 4) {
        b_p0_reg = B(p, 0);
        c_00_reg += b_p0_reg * (*ap0_ptr++);
        c_01_reg += b_p0_reg * (*ap1_ptr++);
        c_02_reg += b_p0_reg * (*ap2_ptr++);
        c_03_reg += b_p0_reg * (*ap3_ptr++);

        b_p0_reg = B(p + 1, 0);
        c_00_reg += b_p0_reg * (*ap0_ptr++);
        c_01_reg += b_p0_reg * (*ap1_ptr++);
        c_02_reg += b_p0_reg * (*ap2_ptr++);
        c_03_reg += b_p0_reg * (*ap3_ptr++);

        b_p0_reg = B(p + 2, 0);
        c_00_reg += b_p0_reg * (*ap0_ptr++);
        c_01_reg += b_p0_reg * (*ap1_ptr++);
        c_02_reg += b_p0_reg * (*ap2_ptr++);
        c_03_reg += b_p0_reg * (*ap3_ptr++);

        b_p0_reg = B(p + 3, 0);
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

void my_matmul_1x4_8(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i += 4) { AddDot1x4_8(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

// loop unrolling
void AddDot1x4_9(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg;// hold C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    float b_p0_reg;                              // holds B(p, 0)

    c_00_reg = 0;
    c_01_reg = 0;
    c_02_reg = 0;
    c_03_reg = 0;

    const float* ap0_ptr = &A(0, 0);
    const float* ap1_ptr = &A(1, 0);
    const float* ap2_ptr = &A(2, 0);
    const float* ap3_ptr = &A(3, 0);

    for (int p = 0; p < k; p += 4) {
        b_p0_reg = B(p, 0);
        c_00_reg += b_p0_reg * (*ap0_ptr);
        c_01_reg += b_p0_reg * (*ap1_ptr);
        c_02_reg += b_p0_reg * (*ap2_ptr);
        c_03_reg += b_p0_reg * (*ap3_ptr);

        b_p0_reg = B(p + 1, 0);
        c_00_reg += b_p0_reg * *(ap0_ptr + 1);
        c_01_reg += b_p0_reg * *(ap1_ptr + 1);
        c_02_reg += b_p0_reg * *(ap2_ptr + 1);
        c_03_reg += b_p0_reg * *(ap3_ptr + 1);

        b_p0_reg = B(p + 2, 0);
        c_00_reg += b_p0_reg * *(ap0_ptr + 2);
        c_01_reg += b_p0_reg * *(ap1_ptr + 2);
        c_02_reg += b_p0_reg * *(ap2_ptr + 2);
        c_03_reg += b_p0_reg * *(ap3_ptr + 2);

        b_p0_reg = B(p + 3, 0);
        c_00_reg += b_p0_reg * *(ap0_ptr + 3);
        c_01_reg += b_p0_reg * *(ap1_ptr + 3);
        c_02_reg += b_p0_reg * *(ap2_ptr + 3);
        c_03_reg += b_p0_reg * *(ap3_ptr + 3);

        ap0_ptr += 4;
        ap1_ptr += 4;
        ap2_ptr += 4;
        ap3_ptr += 4;
    }

    C(0, 0) += c_00_reg;
    C(1, 0) += c_01_reg;
    C(2, 0) += c_02_reg;
    C(3, 0) += c_03_reg;
}

void my_matmul_1x4_9(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i += 4) { AddDot1x4_9(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_3(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    // first row
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));

    // second row
    AddDot(k, &A(1, 0), lda, &B(0, 0), &C(1, 0));
    AddDot(k, &A(1, 0), lda, &B(0, 1), &C(1, 1));
    AddDot(k, &A(1, 0), lda, &B(0, 2), &C(1, 2));
    AddDot(k, &A(1, 0), lda, &B(0, 3), &C(1, 3));

    // third row
    AddDot(k, &A(2, 0), lda, &B(0, 0), &C(2, 0));
    AddDot(k, &A(2, 0), lda, &B(0, 1), &C(2, 1));
    AddDot(k, &A(2, 0), lda, &B(0, 2), &C(2, 2));
    AddDot(k, &A(2, 0), lda, &B(0, 3), &C(2, 3));

    // forth row
    AddDot(k, &A(3, 0), lda, &B(0, 0), &C(3, 0));
    AddDot(k, &A(3, 0), lda, &B(0, 1), &C(3, 1));
    AddDot(k, &A(3, 0), lda, &B(0, 2), &C(3, 2));
    AddDot(k, &A(3, 0), lda, &B(0, 3), &C(3, 3));
}

void my_matmul_4x4_3(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_3(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_4(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    // first row
    for (int p = 0; p < k; p++) { C(0, 0) += A(0, p) * B(p, 0); }

    for (int p = 0; p < k; p++) { C(0, 1) += A(0, p) * B(p, 1); }

    for (int p = 0; p < k; p++) { C(0, 2) += A(0, p) * B(p, 2); }

    for (int p = 0; p < k; p++) { C(0, 3) += A(0, p) * B(p, 3); }

    // second row
    for (int p = 0; p < k; p++) { C(1, 0) += A(1, p) * B(p, 0); }

    for (int p = 0; p < k; p++) { C(1, 1) += A(1, p) * B(p, 1); }

    for (int p = 0; p < k; p++) { C(1, 2) += A(1, p) * B(p, 2); }

    for (int p = 0; p < k; p++) { C(1, 3) += A(1, p) * B(p, 3); }

    // third row
    for (int p = 0; p < k; p++) { C(2, 0) += A(2, p) * B(p, 0); }

    for (int p = 0; p < k; p++) { C(2, 1) += A(2, p) * B(p, 1); }

    for (int p = 0; p < k; p++) { C(2, 2) += A(2, p) * B(p, 2); }

    for (int p = 0; p < k; p++) { C(2, 3) += A(2, p) * B(p, 3); }

    // forth row
    for (int p = 0; p < k; p++) { C(3, 0) += A(3, p) * B(p, 0); }

    for (int p = 0; p < k; p++) { C(3, 1) += A(3, p) * B(p, 1); }

    for (int p = 0; p < k; p++) { C(3, 2) += A(3, p) * B(p, 2); }

    for (int p = 0; p < k; p++) { C(3, 3) += A(3, p) * B(p, 3); }
}

void my_matmul_4x4_4(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_5(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    for (int p = 0; p < k; p++) {
        // first row
        C(0, 0) += A(0, p) * B(p, 0);
        C(0, 1) += A(0, p) * B(p, 1);
        C(0, 2) += A(0, p) * B(p, 2);
        C(0, 3) += A(0, p) * B(p, 3);

        // second row
        C(1, 0) += A(1, p) * B(p, 0);
        C(1, 1) += A(1, p) * B(p, 1);
        C(1, 2) += A(1, p) * B(p, 2);
        C(1, 3) += A(1, p) * B(p, 3);

        // third row
        C(2, 0) += A(2, p) * B(p, 0);
        C(2, 1) += A(2, p) * B(p, 1);
        C(2, 2) += A(2, p) * B(p, 2);
        C(2, 3) += A(2, p) * B(p, 3);

        // forth row
        C(3, 0) += A(3, p) * B(p, 0);
        C(3, 1) += A(3, p) * B(p, 1);
        C(3, 2) += A(3, p) * B(p, 2);
        C(3, 3) += A(3, p) * B(p, 3);
    }
}

void my_matmul_4x4_5(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_5(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_6(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg,
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

    for (int p = 0; p < k; p++) {
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        // first row
        c_00_reg += a_0p_reg * B(p, 0);
        c_01_reg += a_0p_reg * B(p, 1);
        c_02_reg += a_0p_reg * B(p, 2);
        c_03_reg += a_0p_reg * B(p, 3);

        // second row
        c_10_reg += a_1p_reg * B(p, 0);
        c_11_reg += a_1p_reg * B(p, 1);
        c_12_reg += a_1p_reg * B(p, 2);
        c_13_reg += a_1p_reg * B(p, 3);

        // third row
        c_20_reg += a_2p_reg * B(p, 0);
        c_21_reg += a_2p_reg * B(p, 1);
        c_22_reg += a_2p_reg * B(p, 2);
        c_23_reg += a_2p_reg * B(p, 3);

        // forth row
        c_30_reg += a_3p_reg * B(p, 0);
        c_31_reg += a_3p_reg * B(p, 1);
        c_32_reg += a_3p_reg * B(p, 2);
        c_33_reg += a_3p_reg * B(p, 3);
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

void my_matmul_4x4_6(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_6(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_7(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg,
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

void my_matmul_4x4_7(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_7(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_10(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    const float* a_0p_ptr = &A(0, 0);
    const float* a_1p_ptr = &A(1, 0);
    const float* a_2p_ptr = &A(2, 0);
    const float* a_3p_ptr = &A(3, 0);

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
        b_reg.v = _mm_load_ps((float*) &B(p, 0));
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

void my_matmul_4x4_10(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_10(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void AddDot4x4_11(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    const float* a_0p_ptr = &A(0, 0);
    const float* a_1p_ptr = &A(1, 0);
    const float* a_2p_ptr = &A(2, 0);
    const float* a_3p_ptr = &A(3, 0);

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
        b_reg.v = _mm_load_ps((float*) &B(p, 0));
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

void InnerKernel(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) { AddDot4x4_11(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc); }
    }
}

void my_matmul_4x4_11(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    int pb, ib;
    for (int p = 0; p < k; p += KC) {
        pb = std::min(k - p, KC);
        for (int i = 0; i < m; i += MC) {
            ib = std::min(m - i, MC);
            InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
        }
    }
}

void PackMatrixA(int k, float* a, int lda, float* a_to) {
    float* a_0p_ptr = a;
    float* a_1p_ptr = a + lda;
    float* a_2p_ptr = a + (lda << 1);
    float* a_3p_ptr = a + lda * 3;
    for (int p = 0; p < k; p++) {
        *a_to++ = *a_0p_ptr++;
        *a_to++ = *a_1p_ptr++;
        *a_to++ = *a_2p_ptr++;
        *a_to++ = *a_3p_ptr++;
    }
}

void PackMatrixB(int k, float* b, int ldb, float* b_to) {
    for (int p = 0; p < k; p++) {
        float* b_ij_ptr = &B(p, 0);
        *b_to++ = *b_ij_ptr++;
        *b_to++ = *b_ij_ptr++;
        *b_to++ = *b_ij_ptr++;
        *b_to++ = *b_ij_ptr++;
    }
}

void AddDot4x4_13(int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
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
        b_reg.v = _mm_load_ps(b);
        b += 4;

        a_0p_reg.v = _mm_set_ps1(*a++);
        a_1p_reg.v = _mm_set_ps1(*a++);
        a_2p_reg.v = _mm_set_ps1(*a++);
        a_3p_reg.v = _mm_set_ps1(*a++);

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

void InnerKernel_13(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    float packedA[m * k];
    float packedB[k * n];
    for (int j = 0; j < n; j += 4) {
        PackMatrixB(k, &B(0, j), ldb, packedB + j * k);
        for (int i = 0; i < m; i += 4) {
            if (j == 0) { PackMatrixA(k, &A(i, 0), lda, packedA + i * k); }
            AddDot4x4_13(k, packedA + i * k, k, packedB + j * k, 4, &C(i, j), ldc);
        }
    }
}

void my_matmul_4x4_13(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    int pb, ib;
    for (int p = 0; p < k; p += KC) {
        pb = std::min(k - p, KC);
        //#pragma omp parallel for
        for (int i = 0; i < m; i += MC) {
            ib = std::min(m - i, MC);
            InnerKernel_13(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
        }
    }
}