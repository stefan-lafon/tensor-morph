add_executable(tensormorph-opt
  tensormorph-opt.cpp
)

target_link_libraries(tensormorph-opt
  PRIVATE
  TensorMorphPasses
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRIR
  MLIRSupport
  # Libraries for the registered dialects
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRMemRefDialect
)