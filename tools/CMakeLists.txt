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
  # Dialect Libraries
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTosaDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRBufferizationDialect
)