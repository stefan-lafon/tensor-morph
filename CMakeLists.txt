cmake_minimum_required(VERSION 3.15)
project(tensormorph LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLIR/LLVM (using v18 for Colab)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

# Log paths for debugging
message(STATUS "MLIR_DIR: ${MLIR_DIR}")
message(STATUS "LLVM_DIR: ${LLVM_DIR}")

# To be enabled once source files are added
# add_subdirectory(lib)
# add_subdirectory(tools)