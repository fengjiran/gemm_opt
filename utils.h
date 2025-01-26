//
// Created by richard on 5/8/23.
//

#ifndef GEMM_OPT_UTILS_H
#define GEMM_OPT_UTILS_H

#include <iostream>
#include <omp.h>
#include <random>
#include <type_traits>

#include <mmintrin.h>  // MMX
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

// Create macro to let Y(i) equal the ith element of x
#define Y(i) y[(i) * incx]

#define min(i, j) ((i) < (j) ? (i) : (j))
#define max(i, j) ((i) > (j) ? (i) : (j))

// block size
#define MC 256
#define KC 128

typedef union {
    __m128 v;
    float d[4];
} v2df_t;

#endif //GEMM_OPT_UTILS_H
