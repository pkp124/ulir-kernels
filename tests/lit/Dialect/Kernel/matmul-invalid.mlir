// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @matmul_lhs_not_2d(%a: tensor<64xf32>, %b: tensor<64x128xf32>) {
  // expected-error @+1 {{'ks.matmul' op left operand must be a 2D tensor, got rank 1}}
  %0 = ks.matmul %a, %b : tensor<64xf32>, tensor<64x128xf32> -> tensor<64x128xf32>
  return
}

// -----

func.func @matmul_rhs_not_2d(%a: tensor<64x128xf32>, %b: tensor<128xf32>) {
  // expected-error @+1 {{'ks.matmul' op right operand must be a 2D tensor, got rank 1}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<128xf32> -> tensor<64x128xf32>
  return
}

// -----

func.func @matmul_type_mismatch(%a: tensor<64x128xf32>, %b: tensor<128x256xf16>) {
  // expected-error @+1 {{'ks.matmul' op operand element types must match}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<128x256xf16> -> tensor<64x256xf32>
  return
}

// -----

func.func @matmul_inner_dim_mismatch(%a: tensor<64x128xf32>, %b: tensor<64x256xf32>) {
  // expected-error @+1 {{'ks.matmul' op inner dimensions must match: lhs has 128, rhs has 64}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<64x256xf32> -> tensor<64x256xf32>
  return
}

// -----

func.func @matmul_acc_type_too_narrow(%a: tensor<64x128xf32>, %b: tensor<128x256xf32>) {
  // expected-error @+1 {{'ks.matmul' op acc_type (f16) must be at least as wide as input type (f32)}}
  %0 = ks.matmul %a, %b {acc_type = f16} : tensor<64x128xf32>, tensor<128x256xf32> -> tensor<64x256xf32>
  return
}
