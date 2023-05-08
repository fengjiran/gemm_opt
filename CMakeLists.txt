cmake_minimum_required(VERSION 3.25)
project(gemm_opt)

set(CMAKE_CXX_STANDARD 14)

add_executable(gemm_opt main.cpp dclock.h matmul_origin.h
        matmul_1x4_3.h matmul_1x4_4.h matmul_1x4_5.h
        matmul_1x4_6.h matmul_1x4_7.h matmul_1x4_8.h matmul_1x4_9.h matmul_4x4_3.h matmul_4x4_4.h matmul_4x4_5.h)
