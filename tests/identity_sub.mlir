// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Tiny Pattern: x - 0.0 => x
func.func @test_identity_sub(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  
  // CHECK-NOT: tosa.sub
  %0 = "tosa.sub"(%arg0, %zero) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  
  return %0 : tensor<1x4x4x1xf32>
}