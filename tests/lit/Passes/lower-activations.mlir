// RUN: %ks-opt %s --ks-lower-activations | %FileCheck %s

// ============================================================
// ReLU lowering: ks.relu -> arith.maximumf(x, 0)
// ============================================================

// CHECK-LABEL: func @test_relu
// CHECK-NOT: ks.relu
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK: linalg.yield
func.func @test_relu(%input: tensor<32xf32>) -> tensor<32xf32> {
  %output = ks.relu %input : tensor<32xf32>
  return %output : tensor<32xf32>
}

// CHECK-LABEL: func @test_relu_2d
// CHECK-NOT: ks.relu
// CHECK: linalg.generic
// CHECK: arith.maximumf
func.func @test_relu_2d(%input: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %output = ks.relu %input : tensor<8x16xf32>
  return %output : tensor<8x16xf32>
}

// ============================================================
// GELU lowering: ks.gelu -> 0.5 * x * (1 + erf(x / sqrt(2)))
// ============================================================

// CHECK-LABEL: func @test_gelu
// CHECK-NOT: ks.gelu
// CHECK: linalg.generic
// CHECK: math.erf
// CHECK: linalg.yield
func.func @test_gelu(%input: tensor<64xf32>) -> tensor<64xf32> {
  %output = ks.gelu %input : tensor<64xf32>
  return %output : tensor<64xf32>
}

// CHECK-LABEL: func @test_gelu_3d
// CHECK-NOT: ks.gelu
// CHECK: linalg.generic
// CHECK: math.erf
func.func @test_gelu_3d(%input: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %output = ks.gelu %input : tensor<2x4x8xf32>
  return %output : tensor<2x4x8xf32>
}

// ============================================================
// SiLU lowering: ks.silu -> x / (1 + exp(-x))
// ============================================================

// CHECK-LABEL: func @test_silu
// CHECK-NOT: ks.silu
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: linalg.yield
func.func @test_silu(%input: tensor<128xf32>) -> tensor<128xf32> {
  %output = ks.silu %input : tensor<128xf32>
  return %output : tensor<128xf32>
}

// ============================================================
// f16 type variant
// ============================================================

// CHECK-LABEL: func @test_relu_f16
// CHECK-NOT: ks.relu
// CHECK: linalg.generic
// CHECK: arith.maximumf
func.func @test_relu_f16(%input: tensor<32xf16>) -> tensor<32xf16> {
  %output = ks.relu %input : tensor<32xf16>
  return %output : tensor<32xf16>
}

// ============================================================
// Dynamic dimensions
// ============================================================

// CHECK-LABEL: func @test_relu_dynamic
// CHECK-NOT: ks.relu
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic
func.func @test_relu_dynamic(%input: tensor<?xf32>) -> tensor<?xf32> {
  %output = ks.relu %input : tensor<?xf32>
  return %output : tensor<?xf32>
}

// ============================================================
// Chained activations (both should be lowered)
// ============================================================

// CHECK-LABEL: func @test_chain
// CHECK-NOT: ks.relu
// CHECK-NOT: ks.silu
// CHECK-COUNT-2: linalg.generic
func.func @test_chain(%input: tensor<32xf32>) -> tensor<32xf32> {
  %relu = ks.relu %input : tensor<32xf32>
  %silu = ks.silu %relu : tensor<32xf32>
  return %silu : tensor<32xf32>
}
