// RUN: %ks-opt %s --ks-pack | %FileCheck %s
// RUN: %ks-opt %s "--ks-pack=pack-factor=0" | %FileCheck %s --check-prefix=NOOP

// ============================================================
// Basic B packing: [K, N] -> [N/NR, K, NR]
// N=128 divisible by pack-factor=32 (default).
// ============================================================

// CHECK-LABEL: func @test_pack_matmul
// CHECK-NOT:   linalg.matmul
// CHECK:       linalg.pack
// CHECK:       linalg.generic
// CHECK:       linalg.unpack
func.func @test_pack_matmul(
    %A: tensor<64x256xf32>,
    %B: tensor<256x128xf32>,
    %C: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %D = linalg.matmul ins(%A, %B : tensor<64x256xf32>, tensor<256x128xf32>)
                     outs(%C : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %D : tensor<64x128xf32>
}

// ============================================================
// Verify linalg.pack output shape: [N/NR, K, NR] = [4, 256, 32]
// ============================================================

// CHECK-LABEL: func @test_pack_shape
// CHECK:       linalg.pack
// CHECK-SAME:  inner_dims_pos = [1]
// CHECK-SAME:  inner_tiles = [32]
// CHECK-SAME:  tensor<256x128xf32> into tensor<4x256x32xf32>
func.func @test_pack_shape(
    %A: tensor<32x256xf32>,
    %B: tensor<256x128xf32>,
    %C: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %D = linalg.matmul ins(%A, %B : tensor<32x256xf32>, tensor<256x128xf32>)
                     outs(%C : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %D : tensor<32x128xf32>
}

// ============================================================
// pack-factor=0 must be a no-op (linalg.matmul unchanged)
// ============================================================

// NOOP-LABEL: func @test_pack_noop
// NOOP:       linalg.matmul
// NOOP-NOT:   linalg.pack
func.func @test_pack_noop(
    %A: tensor<64x256xf32>,
    %B: tensor<256x128xf32>,
    %C: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %D = linalg.matmul ins(%A, %B : tensor<64x256xf32>, tensor<256x128xf32>)
                     outs(%C : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %D : tensor<64x128xf32>
}

// ============================================================
// Custom pack factor = 8 (for VLEN=256/LMUL=1/f32)
// ============================================================

// RUN: %ks-opt %s "--ks-pack=pack-factor=8" | %FileCheck %s --check-prefix=NR8

// NR8-LABEL: func @test_pack_nr8
// NR8:       linalg.pack
// NR8-SAME:  inner_tiles = [8]
// NR8-SAME:  tensor<256x64xf32> into tensor<8x256x8xf32>
func.func @test_pack_nr8(
    %A: tensor<32x256xf32>,
    %B: tensor<256x64xf32>,
    %C: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %D = linalg.matmul ins(%A, %B : tensor<32x256xf32>, tensor<256x64xf32>)
                     outs(%C : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %D : tensor<32x64xf32>
}
