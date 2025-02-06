//
// Created by richard on 5/8/23.
//

#ifndef GEMM_OPT_MATMUL_H
#define GEMM_OPT_MATMUL_H

#include "utils.h"

static double get_time(timespec* start, timespec* end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}



template<typename T = float,
         typename = std::enable_if_t<std::is_floating_point_v<T>>>
std::vector<T> GenRandomMatrix(size_t m, size_t n, T low = 0, T high = 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<T> matrix(m * n);
    std::uniform_real_distribution<T> dist(low, high);
    for (size_t i = 0; i < m * n; ++i) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

template<typename T,
         typename = void,
         typename = std::enable_if_t<std::is_integral_v<T>>>
std::vector<T> GenRandomMatrix(size_t m, size_t n, T low = 0, T high = 10) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<T> matrix(m * n);
    std::uniform_int_distribution<T> dist(low, high);
    for (size_t i = 0; i < m * n; ++i) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

template<typename T>
float compare_matrix(const std::vector<T>& a, const std::vector<T>& b,
                     size_t m, size_t n, size_t lda, size_t ldb) {
    float max_diff = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float diff = std::abs(a[i * lda + j] - b[i * ldb + j]);
            max_diff += diff;
            if (diff > std::numeric_limits<float>::epsilon()) {
                std::cerr << "error at: (" << i << ", " << j << "): diff = " << diff << std::endl;
            }
        }
    }
    return max_diff;
}

// gemm C = A * B + C
template<typename T>
void matmul_origin(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c,
                   size_t m, size_t n, size_t k,
                   size_t lda, size_t ldb, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T t = 0;
            for (size_t p = 0; p < k; ++p) {
                t += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = t;
        }
    }
}

template<typename T>
void matmul_reorder(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c,
                   size_t m, size_t n, size_t k,
                   size_t lda, size_t ldb, size_t ldc) {
    for (size_t p = 0; p < k; ++p) {
        for (size_t i = 0; i < m; ++i) {
            T t = a[i * lda + p];
            for (size_t j = 0; j < n; ++j) {
                c[i * ldc + j] += t * b[p * ldb + j];
            }
        }
    }
}

void random_matrix(int m, int n, float* a, int lda);

void copy_matrix(int m, int n, const float* a, int lda, float* b, int ldb);

float compare_matrices(int m, int n, float* a, int lda, float* b, int ldb);

void matmul_origin(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_3(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_4(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_5(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_6(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_7(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_8(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_1x4_9(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_3(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_4(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_5(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_6(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_7(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_10(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_11(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

void my_matmul_4x4_13(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc);

#endif//GEMM_OPT_MATMUL_H
