// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: Conv2D -> Mul => Conv2D (with scaled weights and bias)
func.func @test_conv_mul_fusion(%input: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c_weights = "tosa.const"() {value = dense<0.5> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %c_bias    = "tosa.const"() {value = dense<2.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_scale   = "tosa.const"() {value = dense<2.0> : tensor<1xf32>} : () -> tensor<1xf32>

  // 1. Base Conv: (input * 0.5) + 2.0
  %0 = "tosa.conv2d"(%input, %c_weights, %c_bias) {
    dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  // 2. Mul: ((input * 0.5) + 2.0) * 2.0  =>  (input * 1.0) + 4.0
  // CHECK-NOT: tosa.mul
  %result = "tosa.mul"(%0, %c_scale) {shift = 0 : i8} : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  return %result : tensor<1x4x4x1xf32>
}

// CHECK-LABEL: func.func @test_conv_mul_fusion
// CHECK: %[[NEW_WEIGHTS:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1x1xf32>}>
// CHECK: %[[NEW_BIAS:.*]] = "tosa.const"() <{value = dense<4.000000e+00> : tensor<1xf32>}>
// CHECK: tosa.conv2d %arg0, %[[NEW_WEIGHTS]], %[[NEW_BIAS]]