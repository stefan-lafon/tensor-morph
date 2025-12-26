// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

func.func @test_conv_sub_fusion(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
  // CHECK-LABEL: @test_conv_sub_fusion
  %w = "tosa.const"() {value = dense<0.5> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %b = "tosa.const"() {value = dense<4.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %s = "tosa.const"() {value = dense<2.5> : tensor<1xf32>} : () -> tensor<1xf32>

  // CHECK: %[[NEW_B:.*]] = "tosa.const"() <{value = dense<1.500000e+00> : tensor<1xf32>}>
  // CHECK: %[[RES:.*]] = tosa.conv2d %arg0, %{{.*}}, %[[NEW_B]]
  // CHECK-NOT: tosa.sub
  // CHECK: return %[[RES]]
  %0 = "tosa.conv2d"(%arg0, %w, %b) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %1 = "tosa.sub"(%0, %s) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  return %1 : tensor<1x4x4x1xf32>
}