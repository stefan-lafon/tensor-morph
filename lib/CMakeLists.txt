# Define the source files
set(PASS_SOURCES
  Passes/TosaOptimizationsPass.cpp
  Passes/TosaStructuralFusions.cpp
  Passes/TosaAlgebraicFolds.cpp
  Passes/TensorFeatures.cpp
)

# Create the shared library
add_library(TensorMorphPasses SHARED
  ${PASS_SOURCES}
)

target_include_directories(TensorMorphPasses PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib/Passes
  /usr/lib/llvm-18/include
)

# Link TOSA here so the Pattern Matcher can see TosaOps
target_link_libraries(TensorMorphPasses
  PRIVATE
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRMathDialect
  MLIRTosaDialect
  MLIRBufferizationDialect
)

set_target_properties(TensorMorphPasses PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)