// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Testing the "Eating Machine": 
// Conv(w=1, b=0) -> Mul(2) -> Add(3) -> Clamp(0, 6)
// Expected: Conv(w=2, b=3) with fused_activation="clamp", min=0, max=6
func.func @test_mega_fusion(%input: tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32> {
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
  %2 = "tosa.add"(%1, %c_a) : (tensor<1x4x4x1xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  
  // CHECK: tosa.conv2d %arg0, %[[NEW_W]], %[[NEW_B]]
  // CHECK-SAME: clamp_max = 6 : i64
  // CHECK-SAME: clamp_min = 0 : i64
  // CHECK-SAME: fused_activation = "clamp"
  %result = "tosa.clamp"(%2) {min_fp = 0.0 : f32, max_fp = 6.0 : f32, min_int = 0 : i64, max_int = 6 : i64} 
            : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf32>

  return %result : tensor<1x4x4x1xf32>
}