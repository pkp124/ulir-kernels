// RUN: ks-opt %s --ks-lower-to-linalg --ks-tile 2>&1 | FileCheck %s
//
// Tests for the tiling pass applied after lowering to linalg.
// Status: RED (tiling pass not yet implemented)
// Expected behavior: linalg.matmul is tiled into nested scf.for loops.

// CHECK-LABEL: func @tile_square_matmul
// CHECK-NOT: ks.matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
func.func @tile_square_matmul(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: func @tile_large_matmul
// Verify tiling generates loops for larger matrices
// CHECK: scf.for
// CHECK: linalg.matmul
func.func @tile_large_matmul(%A: tensor<256x128xf32>, %B: tensor<128x256xf32>) -> tensor<256x256xf32> {
  %C = ks.matmul %A, %B : tensor<256x128xf32>, tensor<128x256xf32> -> tensor<256x256xf32>
  return %C : tensor<256x256xf32>
}
