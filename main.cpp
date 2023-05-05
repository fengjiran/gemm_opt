#include <iostream>
#include <cstring>
#include <cstdlib>
#include "dclock.h"
#include "matmul_origin.h"
#include "matmul_1x4_3.h"

void random_matrix(int m, int n, float *a, int lda) {
    //double drand48();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A(i, j) = (float) drand48();
        }
    }
}

void copy_matrix(int m, int n, const float *a, int lda, float *b, int ldb) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B(i, j) = A(i, j);
        }
    }
}

float compare_matrices(int m, int n, float *a, int lda, float *b, int ldb) {
    float max_diff = 0.0, diff;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            diff = std::abs(A(i, j) - B(i, j));
            max_diff = std::max(diff, max_diff);
            if (max_diff > 0.5f || max_diff < -0.5f) {
                printf("\n error: i %d  j %d diff %f", i, j, max_diff);
            }
        }
    }
    return max_diff;
}

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main() {
    int m, n, k;
    int lda, ldb, ldc;
    double time_tmp, time_best, gflops, diff;
    float *a, *b, *c, *prec, *nowc;
    struct timespec start, end;
    double time_used = 0.0;

    for (int i = 40; i <= 1000; i += 40) {
        m = i;
        k = i;
        n = i;
        lda = m;
        ldb = k;
        ldc = m;
        gflops = 2.0 * m * n * k * 1.0e-9;
        a = new float[lda * k];
        b = new float[ldb * n];
        c = new float[ldc * n];
        prec = new float[ldc * n];
        nowc = new float[ldc * n];

        // 随机填充矩阵
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);
        random_matrix(m, n, prec, ldc);

        memset(prec, 0, ldc * n * sizeof(float));

        copy_matrix(m, n, prec, ldc, nowc, ldc);

        // 以nowc为基准，判断矩阵运行算结果是否正确
        matmul_origin(m, n, k, a, lda, b, ldb, nowc, ldc);

        // 循环20次，以最快的运行时间为结果
        for (int j = 0; j < 20; j++) {
            copy_matrix(m, n, prec, ldc, c, ldc);
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            my_matmul_1x4_3(m, n, k, a, lda, b, ldb, c, ldc);
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            time_tmp = get_time(&start, &end);

            if (j == 0)
                time_best = time_tmp;
            else
                time_best = std::min(time_best, time_tmp);
        }

        diff = compare_matrices(m, n, c, ldc, nowc, ldc);
        if (diff > 0.5f || diff < -0.5f) {
            exit(0);
        }
        printf("%d %le %le \n", i, gflops / time_best, diff);
        fflush(stdout);

        delete[] a;
        delete[] b;
        delete[] c;
        delete[] prec;
        delete[] nowc;
    }
    std::cout << std::endl;
    fflush(stdout);

    return 0;
}
