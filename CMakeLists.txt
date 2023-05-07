cmake_minimum_required(VERSION 3.25)
project(gemm_opt)

set(CMAKE_CXX_STANDARD 14)

add_executable(gemm_opt main.cpp dclock.h
        utils.h matmul.h matmul.cpp)
