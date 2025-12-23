# Define the source files
set(PASS_SOURCES
  Passes/FoldBatchNorm.cpp
)

# Create the shared library
add_library(TensorMorphPasses SHARED
  ${PASS_SOURCES}
)

# Headers stay the same
target_include_directories(TensorMorphPasses PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  /usr/lib/llvm-18/include
)

# CRITICAL FIX: We remove MLIRSupport and other core libs that cause 
# global variable collisions. We only link what we absolutely need 
# for the compiler to resolve headers.
target_link_libraries(TensorMorphPasses
  PRIVATE
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRMathDialect
)

# Ensure output goes to the right place
set_target_properties(TensorMorphPasses PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)