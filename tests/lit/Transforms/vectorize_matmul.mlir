// RUN: ks-opt %s --ks-lower-to-linalg --ks-tile --ks-vectorize 2>&1 | FileCheck %s
//
// Tests for vectorization pass after tiling.
// Status: RED (vectorize pass not yet implemented)
// Expected: scalar operations become vector.load/fma/store operations.

// CHECK-LABEL: func @vectorize_matmul
// CHECK-NOT: ks.matmul
// After full pipeline, we expect vector operations
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @vectorize_matmul(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: func @vectorize_small_matmul
// Small matrices should also vectorize
// CHECK-NOT: ks.matmul
func.func @vectorize_small_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %C = ks.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %C : tensor<4x4xf32>
}
