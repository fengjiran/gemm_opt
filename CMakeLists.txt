cmake_minimum_required(VERSION 3.25)
project(gemm_opt)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/19.1.7/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/19.1.7/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/Cellar/19.1.7/lib/libomp.dylib")
endif ()

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif ()

add_executable(gemm_opt main.cpp dclock.h
        utils.h matmul.h matmul.cpp)
