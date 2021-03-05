cmake_minimum_required(VERSION 3.2)

SET(CMAKE_CXX_COMPILER clang++)
find_package(LLVM REQUIRED CONFIG) # It doesn't need LLVM exactly, but
				   # uses the bin dir to find OpenMP
				   # and other riscv packages
include_directories(${LLVM_BINARY_DIR}/../sysroot_openmp/include/)

add_library (shmem_decoupling_x86 SHARED src/shmem_decoupling.cpp)

target_include_directories(shmem_decoupling_x86 PUBLIC ${LLVM_BINARY_DIR}/../sysroot_openmp/include/ PUBLIC )
target_compile_options(shmem_decoupling_x86 PUBLIC -std=c++11)

add_library (shmem_decoupling_riscv SHARED src/shmem_decoupling.cpp)

target_include_directories(shmem_decoupling_riscv PUBLIC ${LLVM_BINARY_DIR}/../sysroot_openmp_riscv/include/ PUBLIC )

target_compile_options(shmem_decoupling_riscv PUBLIC -std=c++11 --target=riscv64-unknown-linux-gnu -mabi=lp64d --sysroot=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/riscv64-unknown-linux-gnu/sysroot --gcc-toolchain=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/)

target_link_options(shmem_decoupling_riscv PUBLIC -std=c++11 --target=riscv64-unknown-linux-gnu -mabi=lp64d --sysroot=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/riscv64-unknown-linux-gnu/sysroot --gcc-toolchain=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/)

add_library (maple_decoupling_riscv SHARED src/maple_decoupling.cpp)

target_include_directories(maple_decoupling_riscv PUBLIC ${LLVM_BINARY_DIR}/../sysroot_openmp_riscv/include/ PUBLIC )

target_compile_options(maple_decoupling_riscv PUBLIC -std=c++11 --target=riscv64-unknown-linux-gnu -mabi=lp64d --sysroot=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/riscv64-unknown-linux-gnu/sysroot --gcc-toolchain=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/)

target_link_options(maple_decoupling_riscv PUBLIC -std=c++11 --target=riscv64-unknown-linux-gnu -mabi=lp64d --sysroot=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/riscv64-unknown-linux-gnu/sysroot --gcc-toolchain=${LLVM_BINARY_DIR}/../sysroot_riscv/riscv64-unknown-linux-gnu/)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
file(COPY include/dec_decoupling.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include/)
