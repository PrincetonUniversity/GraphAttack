cmake_minimum_required(VERSION 3.2)

SET(CMAKE_CXX_COMPILER clang++)

find_package(LLVM REQUIRED CONFIG)

add_library (DECADES-numba SHARED DECADES_Numba.cpp)

target_include_directories(DECADES-numba PRIVATE ${LLVM_BINARY_DIR}/../sysroot_openmp/include/)
target_compile_options(DECADES-numba PUBLIC -std=c++11 -fopenmp)

