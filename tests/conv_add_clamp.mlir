// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// This test verifies that a Conv2D followed by an Add and a Clamp 
// is correctly fused into a single Conv2D with bias and activation attributes.

func.func @test_conv_add_clamp_fusion(%input: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c_weights = "tosa.const"() {value = dense<0.5> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %c_bias    = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_add     = "tosa.const"() {value = dense<1.0> : tensor<1xf32>} : () -> tensor<1xf32>

  // 1. Base Convolution
  %0 = "tosa.conv2d"(%input, %c_weights, %c_bias) {
    dilation = array<i64: 1, 1>,
    pad = array<i64: 0, 0, 0, 0>,
    stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  // 2. Add operation (should be folded into bias)
  %1 = "tosa.add"(%0, %c_add) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  // 3. Clamp operation (should be fused as attribute)
  %result = "tosa.clamp"(%1) {
    min_int = 0 : i64,
    max_int = 9223372036854775807 : i64,
    min_fp = 0.0 : f32,
    max_fp = 3.40282347e+38 : f32
  } : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>

  return %result : tensor<1x4x4x1xf32>
}

// CHECK-LABEL: func.func @test_conv_add_clamp_fusion
// CHECK: %[[WEIGHTS:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1x1x1xf32>}>
// CHECK: %[[NEW_BIAS:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1xf32>}>
// CHECK: tosa.conv2d %arg0, %[[WEIGHTS]], %[[NEW_BIAS]]
// CHECK-SAME: clamp_max = 9223372036854775807
// CHECK-SAME: clamp_min = 0
// CHECK-SAME: fused_activation = "clamp"
// CHECK-NOT: tosa.add
// CHECK-NOT: tosa.clamp