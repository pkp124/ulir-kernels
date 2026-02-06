// RUN: ks-opt %s -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// -----
// Valid cases: these should parse and verify without error

// CHECK-LABEL: func @valid_square_f32
func.func @valid_square_f32(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func @valid_rectangular
func.func @valid_rectangular(%A: tensor<32x128xf32>, %B: tensor<128x64xf32>) -> tensor<32x64xf32> {
  %C = ks.matmul %A, %B : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
  return %C : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: func @valid_small_4x4
func.func @valid_small_4x4(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %C = ks.matmul %A, %B : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %C : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @valid_f16
func.func @valid_f16(%A: tensor<16x32xf16>, %B: tensor<32x8xf16>) -> tensor<16x8xf16> {
  %C = ks.matmul %A, %B : tensor<16x32xf16>, tensor<32x8xf16> -> tensor<16x8xf16>
  return %C : tensor<16x8xf16>
}

// -----

// CHECK-LABEL: func @valid_non_power_of_2
func.func @valid_non_power_of_2(%A: tensor<17x23xf32>, %B: tensor<23x31xf32>) -> tensor<17x31xf32> {
  %C = ks.matmul %A, %B : tensor<17x23xf32>, tensor<23x31xf32> -> tensor<17x31xf32>
  return %C : tensor<17x31xf32>
}

// -----

// CHECK-LABEL: func @valid_single_element
func.func @valid_single_element(%A: tensor<1x1xf32>, %B: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %C = ks.matmul %A, %B : tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x1xf32>
  return %C : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func @valid_tall_skinny
func.func @valid_tall_skinny(%A: tensor<256x1xf32>, %B: tensor<1x256xf32>) -> tensor<256x256xf32> {
  %C = ks.matmul %A, %B : tensor<256x1xf32>, tensor<1x256xf32> -> tensor<256x256xf32>
  return %C : tensor<256x256xf32>
}

// -----
// Invalid cases: verifier should catch these

func.func @invalid_1d_lhs(%A: tensor<64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // expected-error @+1 {{left operand must be a 2D tensor}}
  %C = ks.matmul %A, %B : tensor<64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

func.func @invalid_1d_rhs(%A: tensor<64x64xf32>, %B: tensor<64xf32>) -> tensor<64x64xf32> {
  // expected-error @+1 {{right operand must be a 2D tensor}}
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

func.func @invalid_3d_lhs(%A: tensor<2x64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // expected-error @+1 {{left operand must be a 2D tensor}}
  %C = ks.matmul %A, %B : tensor<2x64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// -----

func.func @invalid_inner_dim_mismatch(%A: tensor<32x128xf32>, %B: tensor<64x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{inner dimensions must match}}
  %C = ks.matmul %A, %B : tensor<32x128xf32>, tensor<64x64xf32> -> tensor<32x64xf32>
  return %C : tensor<32x64xf32>
}

// -----

func.func @invalid_type_mismatch(%A: tensor<32x64xf32>, %B: tensor<64x32xf16>) -> tensor<32x32xf32> {
  // expected-error @+1 {{operand element types must match}}
  %C = ks.matmul %A, %B : tensor<32x64xf32>, tensor<64x32xf16> -> tensor<32x32xf32>
  return %C : tensor<32x32xf32>
}
