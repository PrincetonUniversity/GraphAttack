cmake_minimum_required(VERSION 3.2)

find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "Found LLVM: ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if ( NOT (${LLVM_PACKAGE_VERSION} VERSION_EQUAL "11.0"))
  message(FATAL_ERROR "The DECADES compiler requires LLVM version 11.0. If you have this version, then it may be the case that cmake is not finding it. Please set the CMAKE LLVM directory, e.g. using -DLLVM_DIR=~/llvm_build_dir/lib/cmake/llvm in the cmake command")
  endif()

add_subdirectory(AlwaysInliner)
add_subdirectory(DecoupleCompute)
add_subdirectory(DecoupleSupply)
add_subdirectory(ClangVisitor)
add_subdirectory(NumbaFunctionReplace)
add_subdirectory(PrintFunctionCalls)
add_subdirectory(FindNMA)
