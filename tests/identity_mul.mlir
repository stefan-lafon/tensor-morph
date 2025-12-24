// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Tiny Pattern: x * 1.0 => x
func.func @test_identity_mul(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %one = "tosa.const"() {value = dense<1.0> : tensor<1xf32>} : () -> tensor<1xf32>
  
  // CHECK-NOT: tosa.mul
  %0 = "tosa.mul"(%arg0, %one) {shift = 0 : i8} : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  
  return %0 : tensor<1x4x4x1xf32>
}