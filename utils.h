//
// Created by richard on 5/8/23.
//

#ifndef GEMM_OPT_UTILS_H
#define GEMM_OPT_UTILS_H

#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <type_traits>

#include <emmintrin.h>// SSE3
#include <mmintrin.h> // MMX
#include <pmmintrin.h>// SSE2
#include <xmmintrin.h>// SSE

#define STR_CONCAT_(A, B) A##B
#define STR_CONCAT(A, B) STR_CONCAT_(A, B)
#define UNIQUE_ID(prefix) STR_CONCAT(prefix, __COUNTER__)

#define MAX_(t1, t2, name1, name2, x, y) ({ \
    t1 name1 = (x);                         \
    t2 name2 = (y);                         \
    (void) (&name1 == &name2);              \
    name1 > name2 ? name1 : name2;          \
})

#define MAX(x, y) MAX_(decltype(x), decltype(y), UNIQUE_ID(max1_), UNIQUE_ID(max2_), x, y)

#define MIN_(t1, t2, name1, name2, x, y) ({ \
    t1 name1 = (x);                         \
    t2 name2 = (y);                         \
    (void) (&name1 == &name2);              \
    name1 < name2 ? name1 : name2;          \
})

#define MIN(x, y) MIN_(decltype(x), decltype(y), UNIQUE_ID(min1_), UNIQUE_ID(min2_), x, y)

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

// Create macro to let Y(i) equal the ith element of x
#define Y(i) y[(i) * incx]

// #define min(i, j) ((i) < (j) ? (i) : (j))
#define max(i, j) ((i) > (j) ? (i) : (j))

// block size
#define MC 256
#define KC 128

typedef union {
    __m128 v;
    float d[4];
} v2df_t;

class Timer {
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    // Get elapsed time in seconds
    [[nodiscard]] double GetElapsedTime() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) *
               std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

#endif//GEMM_OPT_UTILS_H
