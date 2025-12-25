// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: (x * 2.0) * 3.0 => x * 6.0
func.func @test_mul_chain(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c1 = "tosa.const"() {value = dense<2.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c2 = "tosa.const"() {value = dense<3.0> : tensor<1xf32>} : () -> tensor<1xf32>

  // CHECK: %[[NEW_CONST:.*]] = "tosa.const"() <{value = dense<6.000000e+00> : tensor<1xf32>}>
  
  // CHECK: tosa.mul %arg0, %[[NEW_CONST]] {shift = 0 : i8}
  %0 = "tosa.mul"(%arg0, %c1) {shift = 0 : i8} : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %1 = "tosa.mul"(%0, %c2) {shift = 0 : i8} : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  
  return %1 : tensor<1x4x4x1xf32>
}