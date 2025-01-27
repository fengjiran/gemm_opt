#include <cstring>
#include <cstdlib>
//#include "dclock.h"
#include "matmul.h"

int main() {
    double time_best;
    struct timespec start, end;
    auto x = GenRandomMatrix<float>(5, 5);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << x[i * 5 + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;


    for (int i = 40; i <= 1000; i += 40) {
        int m = i;
        int k = i;
        int n = i;
        int lda = k;
        int ldb = n;
        int ldc = n;
        double gflops = 2.0 * m * n * k * 1.0e-9;
        auto *a = new float[lda * m];
        auto *b = new float[ldb * k];
        auto *c = new float[ldc * m];
        auto *prec = new float[ldc * m];
        auto *nowc = new float[ldc * m];

        // 随机填充矩阵
        random_matrix(m, k, a, lda);
        random_matrix(k, n, b, ldb);
        random_matrix(m, n, prec, ldc);

        memset(prec, 0, ldc * m * sizeof(float));

        copy_matrix(m, n, prec, ldc, nowc, ldc);

        // 以nowc为基准，判断矩阵运行算结果是否正确
        matmul_origin(m, n, k, a, lda, b, ldb, nowc, ldc);

        // 循环20次，以最快的运行时间为结果
        for (int j = 0; j < 20; j++) {
            copy_matrix(m, n, prec, ldc, c, ldc);
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
//            my_matmul_1x4_3(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_1x4_4(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_1x4_5(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_1x4_6(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_1x4_7(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_1x4_8(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_1x4_9(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_4x4_3(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_4x4_4(m, n, k, a, lda, b, ldb, c, ldc);
//            my_matmul_4x4_5(m, n, k, a, lda, b, ldb, c, ldc);
            my_matmul_4x4_13(m, n, k, a, lda, b, ldb, c, ldc);
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            double time_tmp = get_time(&start, &end);

            if (j == 0)
                time_best = time_tmp;
            else
                time_best = min(time_best, time_tmp);
        }

        double diff = compare_matrices(m, n, c, ldc, nowc, ldc);
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
