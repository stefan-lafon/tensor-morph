// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Complex Test: MobileNet-style block
// Sequence: Input -> Transpose -> Pad -> Conv2D -> Mul -> Add -> Clamp
func.func @mobile_block_test(%input: tensor<1x224x224x3xf32>) -> tensor<1x224x224x16xf32> {
  %c_perms = "tosa.const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %c_pad = "tosa.const"() {value = dense<[[0,0], [1,1], [1,1], [0,0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %c_w = "tosa.const"() {value = dense<1.0> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
  %c_b = "tosa.const"() {value = dense<0.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %c_mul = "tosa.const"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %c_add = "tosa.const"() {value = dense<0.5> : tensor<16xf32>} : () -> tensor<16xf32>

  // 1. Transpose
  %0 = "tosa.transpose"(%input, %c_perms) : (tensor<1x224x224x3xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
  
  // 2. Pad
  %1 = "tosa.pad"(%0, %c_pad) : (tensor<1x224x224x3xf32>, tensor<4x2xi32>) -> tensor<1x226x226x3xf32>

  // 3. Conv
  %2 = "tosa.conv2d"(%1, %c_w, %c_b) {
    dilation = array<i64: 1, 1>, 
    pad = array<i64: 0, 0, 0, 0>, 
    stride = array<i64: 1, 1>
  } : (tensor<1x226x226x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x224x224x16xf32>

  // 4. Linear Math
  %3 = "tosa.mul"(%2, %c_mul) {shift = 0 : i8} : (tensor<1x224x224x16xf32>, tensor<16xf32>) -> tensor<1x224x224x16xf32>
  %4 = "tosa.add"(%3, %c_add) : (tensor<1x224x224x16xf32>, tensor<16xf32>) -> tensor<1x224x224x16xf32>

  // 5. Clamp (ReLU6)
  %5 = "tosa.clamp"(%4) {min_int = 0 : i64, max_int = 6 : i64, min_fp = 0.0 : f32, max_fp = 6.0 : f32} : (tensor<1x224x224x16xf32>) -> tensor<1x224x224x16xf32>

  // CHECK: tosa.conv2d
  // CHECK-SAME: clamp_max = 6
  // CHECK-SAME: clamp_min = 0
  // CHECK-SAME: fused_activation = "clamp"
  // CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
  // CHECK-NOT: tosa.transpose
  // CHECK-NOT: tosa.pad
  // CHECK-NOT: tosa.mul
  // CHECK-NOT: tosa.add
  // CHECK-NOT: tosa.clamp
  return %5 : tensor<1x224x224x16xf32>
}