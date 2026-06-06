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

func.func @matmul_result_type_mismatch(%a: tensor<64x128xf32>, %b: tensor<128x256xf32>) {
  // expected-error @+1 {{'ks.matmul' op result element type must match operand element type}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<128x256xf32> -> tensor<64x256xf16>
  return
}

// -----

func.func @matmul_inner_dim_mismatch(%a: tensor<64x128xf32>, %b: tensor<64x256xf32>) {
  // expected-error @+1 {{'ks.matmul' op inner dimensions must match: lhs has 128, rhs has 64}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<64x256xf32> -> tensor<64x256xf32>
  return
}

// -----

func.func @matmul_result_m_dim_mismatch(%a: tensor<64x128xf32>, %b: tensor<128x256xf32>) {
  // expected-error @+1 {{'ks.matmul' op result row dimension must match lhs row dimension: result has 32, lhs has 64}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<128x256xf32> -> tensor<32x256xf32>
  return
}

// -----

func.func @matmul_result_n_dim_mismatch(%a: tensor<64x128xf32>, %b: tensor<128x256xf32>) {
  // expected-error @+1 {{'ks.matmul' op result column dimension must match rhs column dimension: result has 128, rhs has 256}}
  %0 = ks.matmul %a, %b : tensor<64x128xf32>, tensor<128x256xf32> -> tensor<64x128xf32>
  return
}
