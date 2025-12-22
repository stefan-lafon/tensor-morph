add_library(TensorMorphPasses SHARED
  Passes/FoldBatchNorm.cpp
)

target_link_libraries(TensorMorphPasses
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)