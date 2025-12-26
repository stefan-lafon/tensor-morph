// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

func.func @test_depthwise_add(%arg0: tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32> {
  // CHECK-LABEL: @test_depthwise_add
  %w = "tosa.const"() {value = dense<1.0> : tensor<3x3x32x1xf32>} : () -> tensor<3x3x32x1xf32>
  %b = "tosa.const"() {value = dense<0.0> : tensor<32xf32>} : () -> tensor<32xf32>
  %c = "tosa.const"() {value = dense<1.5> : tensor<32xf32>} : () -> tensor<32xf32>

  // CHECK: %[[F_B:.*]] = "tosa.const"() <{value = dense<1.500000e+00> : tensor<32xf32>}>
  // CHECK: %[[RES:.*]] = tosa.depthwise_conv2d %arg0, %{{.*}}, %[[F_B]]
  // CHECK-NOT: tosa.add
  %0 = "tosa.depthwise_conv2d"(%arg0, %w, %b) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x16x16x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x16x16x32xf32>
  %1 = "tosa.add"(%0, %c) : (tensor<1x16x16x32xf32>, tensor<32xf32>) -> tensor<1x16x16x32xf32>
  return %1 : tensor<1x16x16x32xf32>
}

func.func @test_depthwise_mul(%arg0: tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32> {
  // CHECK-LABEL: @test_depthwise_mul
  %w = "tosa.const"() {value = dense<2.0> : tensor<1x1x8x1xf32>} : () -> tensor<1x1x8x1xf32>
  %b = "tosa.const"() {value = dense<1.0> : tensor<8xf32>} : () -> tensor<8xf32>
  %s = "tosa.const"() {value = dense<2.0> : tensor<8xf32>} : () -> tensor<8xf32>

  // CHECK: %[[F_W:.*]] = "tosa.const"() <{value = dense<4.000000e+00> : tensor<1x1x8x1xf32>}>
  // CHECK: %[[F_B:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<8xf32>}>
  // CHECK: tosa.depthwise_conv2d %arg0, %[[F_W]], %[[F_B]]
  %0 = "tosa.depthwise_conv2d"(%arg0, %w, %b) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x16x16x8xf32>, tensor<1x1x8x1xf32>, tensor<8xf32>) -> tensor<1x16x16x8xf32>
  %1 = "tosa.mul"(%0, %s) {shift = 0 : i8} : (tensor<1x16x16x8xf32>, tensor<8xf32>) -> tensor<1x16x16x8xf32>
  return %1 : tensor<1x16x16x8xf32>
}

func.func @test_depthwise_relu(%arg0: tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32> {
  // CHECK-LABEL: @test_depthwise_relu
  %w = "tosa.const"() {value = dense<1.0> : tensor<3x3x32x1xf32>} : () -> tensor<3x3x32x1xf32>
  %b = "tosa.const"() {value = dense<0.0> : tensor<32xf32>} : () -> tensor<32xf32>
  // CHECK: tosa.depthwise_conv2d {{.*}} fused_activation = "clamp"
  %0 = "tosa.depthwise_conv2d"(%arg0, %w, %b) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x16x16x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x16x16x32xf32>
  %1 = "tosa.clamp"(%0) {min_int = 0 : i64, max_int = 6 : i64, min_fp = 0.0 : f32, max_fp = 6.0 : f32} : (tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32>
  return %1 : tensor<1x16x16x32xf32>
}