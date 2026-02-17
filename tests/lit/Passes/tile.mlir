// Tests for --ks-tile: tile linalg.matmul into scf.for loops.
//
// RUN: %ks-opt %s --ks-lower-to-linalg \
// RUN:   "--ks-tile=tile-size-m=4 tile-size-n=4 tile-size-k=8" \
// RUN:   | %FileCheck %s
//
// RUN: %ks-opt %s --ks-lower-to-linalg --ks-tile \
// RUN:   | %FileCheck %s --check-prefix=DEFAULT

// ============================================================
// Basic tiling: 3 nested scf.for loops produced
// ============================================================

// CHECK-LABEL:   func @test_basic_tile
// CHECK-NOT:     ks.matmul
// CHECK:         linalg.fill
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK:         tensor.extract_slice
// CHECK:         linalg.matmul
// CHECK:         tensor.insert_slice

// DEFAULT-LABEL: func @test_basic_tile
// DEFAULT-NOT:   ks.matmul
// DEFAULT:       scf.for
// DEFAULT:       linalg.matmul
func.func @test_basic_tile(%A: tensor<16x32xf32>,
                            %B: tensor<32x16xf32>) -> tensor<16x16xf32> {
  %C = ks.matmul %A, %B : tensor<16x32xf32>, tensor<32x16xf32>
                         -> tensor<16x16xf32>
  return %C : tensor<16x16xf32>
}

// ============================================================
// Rectangular matmul: M != N != K
// ============================================================

// CHECK-LABEL:   func @test_rect_tile
// CHECK-NOT:     ks.matmul
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK:         linalg.matmul

// DEFAULT-LABEL: func @test_rect_tile
// DEFAULT-NOT:   ks.matmul
// DEFAULT:       scf.for
// DEFAULT:       linalg.matmul
func.func @test_rect_tile(%A: tensor<64x128xf32>,
                           %B: tensor<128x32xf32>) -> tensor<64x32xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x32xf32>
                         -> tensor<64x32xf32>
  return %C : tensor<64x32xf32>
}

// ============================================================
// Two chained matmuls: both should be tiled
// ============================================================

// CHECK-LABEL:   func @test_two_matmuls
// CHECK-NOT:     ks.matmul
// CHECK-COUNT-2: linalg.fill
// CHECK:         scf.for

// DEFAULT-LABEL: func @test_two_matmuls
// DEFAULT-NOT:   ks.matmul
// DEFAULT:       scf.for
func.func @test_two_matmuls(%A: tensor<16x32xf32>,
                             %B: tensor<32x16xf32>,
                             %D: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %C = ks.matmul %A, %B : tensor<16x32xf32>, tensor<32x16xf32>
                         -> tensor<16x16xf32>
  %E = ks.matmul %C, %D : tensor<16x16xf32>, tensor<16x8xf32>
                         -> tensor<16x8xf32>
  return %E : tensor<16x8xf32>
}

// ============================================================
// f16 element type: tiling is type-agnostic
// ============================================================

// CHECK-LABEL:   func @test_f16_tile
// CHECK-NOT:     ks.matmul
// CHECK:         linalg.fill
// CHECK:         scf.for
// CHECK:         linalg.matmul

// DEFAULT-LABEL: func @test_f16_tile
// DEFAULT-NOT:   ks.matmul
// DEFAULT:       linalg.matmul
func.func @test_f16_tile(%A: tensor<16x32xf16>,
                          %B: tensor<32x16xf16>) -> tensor<16x16xf16> {
  %C = ks.matmul %A, %B : tensor<16x32xf16>, tensor<32x16xf16>
                         -> tensor<16x16xf16>
  return %C : tensor<16x16xf16>
}
