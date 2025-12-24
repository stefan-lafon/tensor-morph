// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: Conv2D -> Sub => Conv2D (with updated bias)
func.func @test_conv_sub_fusion(%input: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c_weights = "tosa.const"() {value = dense<0.5> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %c_bias    = "tosa.const"() {value = dense<4.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_sub     = "tosa.const"() {value = dense<1.5> : tensor<1xf32>} : () -> tensor<1xf32>

  // 1. Base Conv: (input * 0.5) + 4.0
  %0 = "tosa.conv2d"(%input, %c_weights, %c_bias) {
    dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  // 2. Sub: ((input * 0.5) + 4.0) - 1.5  =>  (input * 0.5) + 2.5
  // CHECK-NOT: tosa.sub
  %result = "tosa.sub"(%0, %c_sub) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  return %result : tensor<1x4x4x1xf32>
}

// CHECK-LABEL: func.func @test_conv_sub_fusion
// CHECK: %[[WEIGHTS:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1x1x1xf32>}>
// CHECK: %[[NEW_BIAS:.*]] = "tosa.const"() <{value = dense<2.500000e+00> : tensor<1xf32>}>
// CHECK: tosa.conv2d %arg0, %[[WEIGHTS]], %[[NEW_BIAS]]