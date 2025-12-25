// RUN: tensormorph-opt --tosa-opt %s | FileCheck %s

// Pattern: 
// Input[1, 3, 2, 1] -> Transpose[0, 2, 1, 3] -> Conv2D -> Output
// Expected:
// Input[1, 3, 2, 1] -> Conv2D (with Permuted Weights) -> Output
func.func @test_transpose_fold(%input: tensor<1x3x2x1xf32>) -> tensor<1x3x2x1xf32> {
  %perms = "tosa.const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  
  // Weights for a 2x1 kernel (H=2, W=1)
  %c_w = "tosa.const"() {value = dense<[[[[1.0]], [[2.0]]]]> : tensor<1x2x1x1xf32>} : () -> tensor<1x2x1x1xf32>
  %c_b = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>

  // The Transpose Op that we want to eliminate
  %0 = "tosa.transpose"(%input, %perms) : (tensor<1x3x2x1xf32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>

  // The Convolution Op
  // CHECK-NOT: tosa.transpose
  // CHECK: %[[NEW_W:.*]] = "tosa.const"() <{value = dense<{{.*}}1.000000e+00{{.*}}2.000000e+00{{.*}}> : tensor<1x1x2x1xf32>}>
  // CHECK: tosa.conv2d %arg0, %[[NEW_W]], {{.*}}
  %1 = "tosa.conv2d"(%0, %c_w, %c_b) {
    dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
  } : (tensor<1x2x3x1xf32>, tensor<1x2x1x1xf32>, tensor<1xf32>) -> tensor<1x3x2x1xf32>

  return %1 : tensor<1x3x2x1xf32>
}