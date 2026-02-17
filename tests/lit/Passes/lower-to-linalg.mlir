// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// ============================================================
// Static shapes — basic square matmul
// ============================================================

// CHECK-LABEL: func @test_static_square
// CHECK-NOT:   ks.matmul
// CHECK:       tensor.empty
// CHECK:       arith.constant 0
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
func.func @test_static_square(%A: tensor<32x32xf32>,
                               %B: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %C = ks.matmul %A, %B : tensor<32x32xf32>, tensor<32x32xf32>
                         -> tensor<32x32xf32>
  return %C : tensor<32x32xf32>
}

// ============================================================
// Static shapes — rectangular matmul
// ============================================================

// CHECK-LABEL: func @test_static_rect
// CHECK-NOT:   ks.matmul
// CHECK:       tensor.empty
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
func.func @test_static_rect(%A: tensor<64x128xf32>,
                             %B: tensor<128x256xf32>) -> tensor<64x256xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32>
                         -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}

// ============================================================
// Dynamic M dimension
// ============================================================

// CHECK-LABEL: func @test_dynamic_m
// CHECK-NOT:   ks.matmul
// CHECK:       tensor.dim
// CHECK:       tensor.empty
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
func.func @test_dynamic_m(%A: tensor<?x128xf32>,
                           %B: tensor<128x64xf32>) -> tensor<?x64xf32> {
  %C = ks.matmul %A, %B : tensor<?x128xf32>, tensor<128x64xf32>
                         -> tensor<?x64xf32>
  return %C : tensor<?x64xf32>
}

// ============================================================
// Dynamic N dimension
// ============================================================

// CHECK-LABEL: func @test_dynamic_n
// CHECK-NOT:   ks.matmul
// CHECK:       tensor.dim
// CHECK:       tensor.empty
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
func.func @test_dynamic_n(%A: tensor<64x128xf32>,
                           %B: tensor<128x?xf32>) -> tensor<64x?xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x?xf32>
                         -> tensor<64x?xf32>
  return %C : tensor<64x?xf32>
}

// ============================================================
// Both M and N dynamic
// ============================================================

// CHECK-LABEL: func @test_dynamic_both
// CHECK-NOT:   ks.matmul
// CHECK-COUNT-2: tensor.dim
// CHECK:       tensor.empty
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
func.func @test_dynamic_both(%A: tensor<?x128xf32>,
                              %B: tensor<128x?xf32>) -> tensor<?x?xf32> {
  %C = ks.matmul %A, %B : tensor<?x128xf32>, tensor<128x?xf32>
                         -> tensor<?x?xf32>
  return %C : tensor<?x?xf32>
}

// ============================================================
// f16 element type — zero fill must match element type
// ============================================================

// CHECK-LABEL: func @test_f16
// CHECK-NOT:   ks.matmul
// CHECK:       tensor.empty
// CHECK:       arith.constant 0
// CHECK:       linalg.fill
// CHECK:       linalg.matmul
func.func @test_f16(%A: tensor<16x16xf16>,
                    %B: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %C = ks.matmul %A, %B : tensor<16x16xf16>, tensor<16x16xf16>
                         -> tensor<16x16xf16>
  return %C : tensor<16x16xf16>
}

// ============================================================
// Chained: two matmuls should both be lowered
// ============================================================

// CHECK-LABEL: func @test_chain
// CHECK-NOT:   ks.matmul
// CHECK-COUNT-2: linalg.matmul
func.func @test_chain(%A: tensor<32x64xf32>,
                      %B: tensor<64x32xf32>,
                      %D: tensor<32x16xf32>) -> tensor<32x16xf32> {
  // First matmul: 32x64 * 64x32 → 32x32
  %C = ks.matmul %A, %B : tensor<32x64xf32>, tensor<64x32xf32>
                         -> tensor<32x32xf32>
  // Second matmul: 32x32 * 32x16 → 32x16
  %E = ks.matmul %C, %D : tensor<32x32xf32>, tensor<32x16xf32>
                         -> tensor<32x16xf32>
  return %E : tensor<32x16xf32>
}

// ============================================================
// Verify zero accumulator semantics:
// The fill op must use constant 0, not an arbitrary value.
// ============================================================

// CHECK-LABEL: func @test_zero_init
// CHECK:       %[[Z:.*]] = arith.constant 0.0{{.*}} : f32
// CHECK:       linalg.fill ins(%[[Z]]
func.func @test_zero_init(%A: tensor<8x8xf32>,
                           %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %C = ks.matmul %A, %B : tensor<8x8xf32>, tensor<8x8xf32> -> tensor<8x8xf32>
  return %C : tensor<8x8xf32>
}
