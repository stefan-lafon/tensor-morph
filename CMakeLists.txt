cmake_minimum_required(VERSION 3.10)
project(tensormorph)

set(CMAKE_CXX_STANDARD 17)

# MLIR/LLVM setup
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/lib/Passes)

add_definitions(${LLVM_DEFINITIONS})

# Modular Optimization Library
# We include TensorFeatures.cpp here to satisfy the linker
add_library(TosaOptimizations SHARED
    lib/Passes/TosaOptimizationsPass.cpp
    lib/Passes/TosaStructuralFusions.cpp
    lib/Passes/TosaAlgebraicFolds.cpp
    lib/Passes/TensorFeatures.cpp
)

target_link_libraries(TosaOptimizations
    PUBLIC
    MLIRTosaDialect
    MLIRPass
    MLIRTransforms
    MLIRIR
    MLIRArithDialect
)

# Main tool driver: tensormorph-opt
add_executable(tensormorph-opt tools/tensormorph-opt.cpp)
target_link_libraries(tensormorph-opt
    PRIVATE
    TosaOptimizations
    MLIROptLib
    MLIRTosaDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRPass
    MLIRSupport
    MLIRParser
)