// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

func.func @test_conv_sub_fusion(%input: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  %c_w = "tosa.const"() {value = dense<0.5> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %c_b = "tosa.const"() {value = dense<4.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_s = "tosa.const"() {value = dense<1.5> : tensor<1xf32>} : () -> tensor<1xf32>

  // CHECK: %[[NEW_B:.*]] = "tosa.const"() <{value = dense<2.500000e+00> : tensor<1xf32>}>
  // CHECK: tosa.conv2d %arg0, {{.*}}, %[[NEW_B]]
  %0 = "tosa.conv2d"(%input, %c_w, %c_b) {
    dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  %1 = "tosa.sub"(%0, %c_s) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  return %1 : tensor<1x4x4x1xf32>
}