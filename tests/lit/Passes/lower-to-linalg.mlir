// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// ============================================================
// Basic matmul lowering: ks.matmul -> linalg.matmul
// ============================================================

// CHECK-LABEL: func @test_matmul_basic
// CHECK-NOT: ks.matmul
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64x256xf32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<64x256xf32>)
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<64x128xf32>, tensor<128x256xf32>) outs(%[[FILL]] : tensor<64x256xf32>)
func.func @test_matmul_basic(%A: tensor<64x128xf32>, %B: tensor<128x256xf32>)
    -> tensor<64x256xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32>
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}

// ============================================================
// Square matmul
// ============================================================

// CHECK-LABEL: func @test_matmul_square
// CHECK-NOT: ks.matmul
// CHECK: tensor.empty() : tensor<64x64xf32>
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_matmul_square(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32>
                          -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// ============================================================
// f16 type variant
// ============================================================

// CHECK-LABEL: func @test_matmul_f16
// CHECK-NOT: ks.matmul
// CHECK: %[[E:.*]] = tensor.empty() : tensor<32x128xf16>
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: linalg.fill ins(%[[Z]] : f16) outs(%[[E]] : tensor<32x128xf16>)
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf16>, tensor<64x128xf16>)
func.func @test_matmul_f16(%A: tensor<32x64xf16>, %B: tensor<64x128xf16>)
    -> tensor<32x128xf16> {
  %C = ks.matmul %A, %B : tensor<32x64xf16>, tensor<64x128xf16>
                          -> tensor<32x128xf16>
  return %C : tensor<32x128xf16>
}

// ============================================================
// Dynamic dimensions
// ============================================================

// CHECK-LABEL: func @test_matmul_dynamic
// CHECK-NOT: ks.matmul
// CHECK: tensor.dim
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_matmul_dynamic(%A: tensor<?x128xf32>, %B: tensor<128x?xf32>)
    -> tensor<?x?xf32> {
  %C = ks.matmul %A, %B : tensor<?x128xf32>, tensor<128x?xf32>
                          -> tensor<?x?xf32>
  return %C : tensor<?x?xf32>
}

// ============================================================
// Mixed-precision: f16 inputs with f32 accumulator
// ============================================================

// CHECK-LABEL: func @test_matmul_acc_type
// CHECK-NOT: ks.matmul
// CHECK: tensor.empty() : tensor<32x64xf32>
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: linalg.fill ins(%[[Z]] : f32)
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x128xf16>, tensor<128x64xf16>) outs(%{{.*}} : tensor<32x64xf32>)
// CHECK: linalg.generic
// CHECK:   arith.truncf
// CHECK:   linalg.yield
func.func @test_matmul_acc_type(%A: tensor<32x128xf16>, %B: tensor<128x64xf16>)
    -> tensor<32x64xf16> {
  %C = ks.matmul %A, %B {acc_type = f32}
      : tensor<32x128xf16>, tensor<128x64xf16> -> tensor<32x64xf16>
  return %C : tensor<32x64xf16>
}

// ============================================================
// acc_type matching result type (no truncation needed)
// ============================================================

// CHECK-LABEL: func @test_matmul_acc_type_same
// CHECK-NOT: ks.matmul
// CHECK: linalg.matmul
// CHECK-NOT: arith.truncf
func.func @test_matmul_acc_type_same(%A: tensor<32x64xf32>, %B: tensor<64x128xf32>)
    -> tensor<32x128xf32> {
  %C = ks.matmul %A, %B {acc_type = f32}
      : tensor<32x64xf32>, tensor<64x128xf32> -> tensor<32x128xf32>
  return %C : tensor<32x128xf32>
}

// ============================================================
// Multiple matmuls in same function
// ============================================================

// CHECK-LABEL: func @test_matmul_chain
// CHECK-NOT: ks.matmul
// CHECK-COUNT-2: linalg.matmul
func.func @test_matmul_chain(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>,
                             %D: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x64xf32>
                          -> tensor<64x64xf32>
  %E = ks.matmul %C, %D : tensor<64x64xf32>, tensor<64x32xf32>
                          -> tensor<64x32xf32>
  return %E : tensor<64x32xf32>
}
