// RUN: %ks-opt %s --ks-lower-to-linalg --ks-tile='tile-m=32 tile-n=32 tile-k=16' | %FileCheck %s

// ============================================================
// Tiled matmul: linalg.matmul -> scf.for loops around smaller matmul
// ============================================================

// CHECK-LABEL: func @test_tile_matmul
// CHECK-NOT: ks.matmul
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.matmul
func.func @test_tile_matmul(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32>
                          -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// ============================================================
// Rectangular matmul with tiling
// ============================================================

// CHECK-LABEL: func @test_tile_rectangular
// CHECK-NOT: ks.matmul
// CHECK: scf.for
// CHECK:   linalg.matmul
func.func @test_tile_rectangular(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>)
    -> tensor<128x64xf32> {
  %C = ks.matmul %A, %B : tensor<128x256xf32>, tensor<256x64xf32>
                          -> tensor<128x64xf32>
  return %C : tensor<128x64xf32>
}

// ============================================================
// Default tile sizes (64x64x32 from generic profile)
// ============================================================

// RUN: %ks-opt %s --ks-lower-to-linalg --ks-tile | %FileCheck %s --check-prefix=DEFAULT

// DEFAULT-LABEL: func @test_tile_defaults
// DEFAULT: scf.for
// DEFAULT:   linalg.matmul
func.func @test_tile_defaults(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %C = ks.matmul %A, %B : tensor<128x128xf32>, tensor<128x128xf32>
                          -> tensor<128x128xf32>
  return %C : tensor<128x128xf32>
}
