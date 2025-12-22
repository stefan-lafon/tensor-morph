cmake_minimum_required(VERSION 3.22)
project(TensorMorph)

set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)

add_subdirectory(lib)
add_subdirectory(tools)