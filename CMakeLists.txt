cmake_minimum_required(VERSION 3.25)
project(gemm_opt)

set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif ()

add_executable(gemm_opt main.cpp dclock.h
        utils.h matmul.h matmul.cpp)
