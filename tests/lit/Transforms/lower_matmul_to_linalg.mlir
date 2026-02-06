// RUN: ks-opt %s --ks-lower-to-linalg 2>&1 | FileCheck %s
//
// Tests for the ks.matmul -> linalg.matmul lowering pass.
// Status: RED (pass not yet implemented)
// These tests define the expected behavior of the lowering pass.

// CHECK-LABEL: func @lower_square_matmul
// CHECK-NOT: ks.matmul
// CHECK-DAG: tensor.empty
// CHECK: linalg.matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<64x64xf32>, tensor<64x64xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<64x64xf32>)
func.func @lower_square_matmul(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: func @lower_rect_matmul
// CHECK-NOT: ks.matmul
// CHECK-DAG: tensor.empty() : tensor<32x64xf32>
// CHECK: linalg.matmul
func.func @lower_rect_matmul(%A: tensor<32x128xf32>, %B: tensor<128x64xf32>) -> tensor<32x64xf32> {
  %C = ks.matmul %A, %B : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
  return %C : tensor<32x64xf32>
}

// CHECK-LABEL: func @lower_small_4x4
// CHECK-NOT: ks.matmul
// CHECK: linalg.matmul
func.func @lower_small_4x4(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %C = ks.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %C : tensor<4x4xf32>
}

// CHECK-LABEL: func @lower_f16_matmul
// CHECK-NOT: ks.matmul
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x32xf16>
func.func @lower_f16_matmul(%A: tensor<16x32xf16>, %B: tensor<32x8xf16>) -> tensor<16x8xf16> {
  %C = ks.matmul %A, %B : tensor<16x32xf16>, tensor<32x8xf16> -> tensor<16x8xf16>
  return %C : tensor<16x8xf16>
}

// CHECK-LABEL: func @lower_non_power_of_2
// CHECK-NOT: ks.matmul
// CHECK: linalg.matmul
func.func @lower_non_power_of_2(%A: tensor<17x23xf32>, %B: tensor<23x31xf32>) -> tensor<17x31xf32> {
  %C = ks.matmul %A, %B : tensor<17x23xf32>, tensor<23x31xf32> -> tensor<17x31xf32>
  return %C : tensor<17x31xf32>
}
