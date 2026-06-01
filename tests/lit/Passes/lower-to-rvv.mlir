// RUN: %ks-opt %s --ks-lower-to-linalg --ks-lower-to-rvv | %FileCheck %s

// ============================================================
// After --ks-lower-to-rvv the IR must be in LLVM dialect.
// All func, arith, memref, vector, scf ops must be gone.
// ============================================================

// CHECK-LABEL: llvm.func @test_lower_to_rvv_relu
// CHECK-NOT: func.func
// CHECK-NOT: arith.constant
// CHECK-NOT: vector.transfer
// CHECK:     llvm.return
func.func @test_lower_to_rvv_relu(%arg0: memref<32xf32>,
                                   %arg1: memref<32xf32>) {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %cv = vector.broadcast %zero : f32 to vector<32xf32>
  %v = vector.transfer_read %arg0[%c0], %zero
      : memref<32xf32>, vector<32xf32>
  %r = arith.maximumf %v, %cv : vector<32xf32>
  vector.transfer_write %r, %arg1[%c0]
      : vector<32xf32>, memref<32xf32>
  return
}

// ============================================================
// Verify LLVM dialect function linkage.
// ============================================================

// CHECK-LABEL: llvm.func @test_lower_linkage
// CHECK-SAME:  (
func.func @test_lower_linkage(%arg0: memref<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %v = memref.load %arg0[%c0] : memref<8xf32>
  return %v : f32
}

// ============================================================
// End-to-end lowering pipeline in one invocation:
//   ks.matmul -> linalg -> tile -> vectorize -> rvv
// Packing is covered in pack.mlir; linalg.pack bufferization is tracked
// separately from this LLVM lowering smoke test.
// ============================================================

// RUN: %ks-opt %s \
// RUN:   --ks-lower-to-linalg \
// RUN:   "--ks-tile=tile-size-m=128 tile-size-n=128 tile-size-k=256" \
// RUN:   --ks-vectorize \
// RUN:   --ks-lower-to-rvv \
// RUN:   | %FileCheck %s --check-prefix=E2E

// E2E-LABEL: llvm.func @test_rvv_matmul_e2e
// E2E-NOT:   ks.matmul
// E2E-NOT:   linalg.matmul
// E2E-NOT:   vector.contract
// E2E-SAME:  (
func.func @test_rvv_matmul_e2e(
    %A: memref<128x256xf32>,
    %B: memref<256x128xf32>,
    %C: memref<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %At = bufferization.to_tensor %A restrict : memref<128x256xf32> to tensor<128x256xf32>
  %Bt = bufferization.to_tensor %B restrict : memref<256x128xf32> to tensor<256x128xf32>
  %Ct = bufferization.to_tensor %C restrict writable : memref<128x128xf32> to tensor<128x128xf32>
  %D = ks.matmul %At, %Bt
      : tensor<128x256xf32>, tensor<256x128xf32> -> tensor<128x128xf32>
  %Dm = bufferization.to_buffer %D : tensor<128x128xf32> to memref<128x128xf32>
  memref.copy %Dm, %C : memref<128x128xf32> to memref<128x128xf32>
  return
}
