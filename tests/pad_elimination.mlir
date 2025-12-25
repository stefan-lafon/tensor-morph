// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: 
// Input[1, 3, 3, 1] -> Pad(1,1,1,1) -> Conv2D(pad=1,1,1,1) -> Output
// Expected:
// Input[1, 3, 3, 1] -> Conv2D(pad=2,2,2,2) -> Output
func.func @test_pad_elimination(%input: tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32> {
  // Padding constant: [0,0, 1,1, 1,1, 0,0] for [N, H, W, C]
  %c_pad = "tosa.const"() {value = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  
  %c_w = "tosa.const"() {value = dense<1.0> : tensor<1x3x3x1xf32>} : () -> tensor<1x3x3x1xf32>
  %c_b = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>

  // Explicit padding operation
  %0 = "tosa.pad"(%input, %c_pad) : (tensor<1x3x3x1xf32>, tensor<4x2xi32>) -> tensor<1x5x5x1xf32>

  // CHECK-NOT: tosa.pad
  // CHECK: tosa.conv2d %arg0, {{.*}} pad = array<i64: 2, 2, 2, 2>
  %1 = "tosa.conv2d"(%0, %c_w, %c_b) {
    dilation = array<i64: 1, 1>, 
    pad = array<i64: 1, 1, 1, 1>, 
    stride = array<i64: 1, 1>
  } : (tensor<1x5x5x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>) -> tensor<1x3x3x1xf32>

  return %1 : tensor<1x3x3x1xf32>
}