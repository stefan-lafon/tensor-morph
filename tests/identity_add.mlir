// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Simple sanity check: Ensure adding a constant zero is optimized out.

func.func @test_identity_add(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %zero = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  
  // CHECK-NOT: tosa.add
  %0 = "tosa.add"(%arg0, %zero) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  
  return %0 : tensor<1x4x4x1xf32>
}