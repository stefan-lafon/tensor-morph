// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

func.func @test_fanout_cloning(%input: tensor<1x4x4x1xf32>) -> (tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>) {
  %c_w = "tosa.const"() {value = dense<1.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %c_b = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  
  %c_add1 = "tosa.const"() {value = dense<1.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %c_add2 = "tosa.const"() {value = dense<2.0> : tensor<1xf32>} : () -> tensor<1xf32>

  %0 = "tosa.conv2d"(%input, %c_w, %c_b) {
    dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  %res1 = "tosa.add"(%0, %c_add1) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %res2 = "tosa.add"(%0, %c_add2) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>

  // CHECK-DAG: %[[C1:.*]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1xf32>}>
  // CHECK-DAG: %[[C2:.*]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<1xf32>}>
  
  // Verify we have two separate convolutions using these constants as their third operand (bias)
  // CHECK-DAG: tosa.conv2d %arg0, {{.*}}, %[[C1]]
  // CHECK-DAG: tosa.conv2d %arg0, {{.*}}, %[[C2]]

  return %res1, %res2 : tensor<1x4x4x1xf32>, tensor<1x4x4x1xf32>
}