cmake_minimum_required(VERSION 3.25)
project(gemm_opt)

set(CMAKE_CXX_STANDARD 17)

add_executable(gemm_opt main.cpp dclock.h matmul_origin.h matmul_1x4_3.h matmul_1x4_4.h)
