// RUN: ks-opt %s --ks-lower-to-linalg --ks-tile --ks-vectorize --ks-lower-to-rvv 2>&1 | FileCheck %s
//
// Tests for the full pipeline from ks.matmul to RVV intrinsics.
// Status: RED (RVV lowering pass not yet implemented)
// Expected: vector operations become LLVM calls to RISC-V vector intrinsics.

// CHECK-LABEL: func @matmul_to_rvv
// CHECK-NOT: ks.matmul
// CHECK-NOT: linalg.matmul
// After full lowering, expect RVV-style operations:
// CHECK: llvm.call @llvm.riscv
func.func @matmul_to_rvv(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: func @matmul_4x4_to_rvv
// Small matmul should also lower completely
// CHECK-NOT: ks.matmul
func.func @matmul_4x4_to_rvv(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %C = ks.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %C : tensor<4x4xf32>
}
