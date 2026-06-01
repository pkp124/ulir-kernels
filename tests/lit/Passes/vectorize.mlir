// RUN: %ks-opt %s --ks-vectorize | %FileCheck %s

// ============================================================
// Vectorize linalg.matmul -> vector operations + transfer ops
// Tile sizes must be static (set before by --ks-tile).
// ============================================================

// CHECK-LABEL: func @test_vectorize_matmul
// CHECK-NOT:   linalg.matmul
// CHECK:       vector.transfer_read
// CHECK:       vector.multi_reduction
// CHECK:       vector.transfer_write
// CHECK-NOT:   linalg.matmul
func.func @test_vectorize_matmul(
    %A: tensor<16x32xf32>,
    %B: tensor<32x32xf32>,
    %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %D = linalg.matmul ins(%A, %B : tensor<16x32xf32>, tensor<32x32xf32>)
                     outs(%C : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %D : tensor<16x32xf32>
}

// ============================================================
// Vectorize linalg.generic (e.g., from activation lowering)
// ============================================================

// CHECK-LABEL: func @test_vectorize_generic
// CHECK-NOT:   linalg.generic
// CHECK:       vector.transfer_read
// CHECK:       vector.transfer_write
// CHECK-NOT:   linalg.generic
func.func @test_vectorize_generic(%input: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %zero = arith.constant 0.0 : f32
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<32xf32>)
      outs(%empty : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.maximumf %in, %zero : f32
      linalg.yield %v : f32
  } -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// ============================================================
// Full M4 pipeline: lower-to-linalg -> tile -> vectorize
// ============================================================

// RUN: %ks-opt %s \
// RUN:   --ks-lower-to-linalg \
// RUN:   "--ks-tile=tile-size-m=16 tile-size-n=32 tile-size-k=32" \
// RUN:   --ks-vectorize \
// RUN:   | %FileCheck %s --check-prefix=PIPELINE

// PIPELINE-LABEL: func @test_pipeline_vectorize
// PIPELINE-NOT:   ks.matmul
// PIPELINE-NOT:   linalg.matmul
// PIPELINE:       vector.transfer_read
// PIPELINE:       vector.multi_reduction
// PIPELINE-NOT:   linalg.matmul
func.func @test_pipeline_vectorize(
    %A: tensor<128x256xf32>,
    %B: tensor<256x128xf32>) -> tensor<128x128xf32> {
  %C = ks.matmul %A, %B : tensor<128x256xf32>, tensor<256x128xf32> -> tensor<128x128xf32>
  return %C : tensor<128x128xf32>
}
