// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: Conv -> Mul(2.0) -> Add(3.0) 
// Should result in: Conv (weights * 2.0, bias * 2.0 + 3.0)
func.func @test_conv_affine_fusion(%input: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c_w = "tosa.const"() {value = dense<1.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %c_b = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_m = "tosa.const"() {value = dense<2.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_a = "tosa.const"() {value = dense<3.0> : tensor<1xf32>} : () -> tensor<1xf32>

  // CHECK: %[[NEW_W:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<1x1x1x1xf32>}>
  // CHECK: %[[NEW_B:.*]] = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1xf32>}>
  
  %0 = "tosa.conv2d"(%input, %c_w, %c_b) {
    dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  %1 = "tosa.mul"(%0, %c_m) {shift = 0 : i8} : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %result = "tosa.add"(%1, %c_a) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  // CHECK-NOT: tosa.mul
  // CHECK-NOT: tosa.add
  return %result : tensor<1x4x4x1xf32>
}