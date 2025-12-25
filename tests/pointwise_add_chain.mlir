// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: (x + 1.5) + 2.5 => x + 4.0
func.func @test_add_chain(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c1 = "tosa.const"() {value = dense<1.5> : tensor<1xf32>} : () -> tensor<1xf32>
  %c2 = "tosa.const"() {value = dense<2.5> : tensor<1xf32>} : () -> tensor<1xf32>

  // CHECK: %[[NEW_CONST:.*]] = "tosa.const"() <{value = dense<4.000000e+00> : tensor<1xf32>}>
  
  // The first add should be deleted. The second becomes a single tosa.add
  // CHECK: tosa.add %arg0, %[[NEW_CONST]]
  %0 = "tosa.add"(%arg0, %c1) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %1 = "tosa.add"(%0, %c2) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  
  return %1 : tensor<1x4x4x1xf32>
}