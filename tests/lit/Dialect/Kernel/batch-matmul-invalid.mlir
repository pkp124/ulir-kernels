// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @batch_matmul_lhs_2d(%a: tensor<64x128xf32>, %b: tensor<2x128x256xf32>) {
  // expected-error @+1 {{'ks.batch_matmul' op left operand must have at least 3 dimensions}}
  %0 = ks.batch_matmul %a, %b : tensor<64x128xf32>, tensor<2x128x256xf32> -> tensor<2x64x256xf32>
  return
}

// -----

func.func @batch_matmul_rhs_2d(%a: tensor<2x64x128xf32>, %b: tensor<128x256xf32>) {
  // expected-error @+1 {{'ks.batch_matmul' op right operand must have at least 3 dimensions}}
  %0 = ks.batch_matmul %a, %b : tensor<2x64x128xf32>, tensor<128x256xf32> -> tensor<2x64x256xf32>
  return
}

// -----

func.func @batch_matmul_type_mismatch(%a: tensor<2x64x128xf32>, %b: tensor<2x128x256xf16>) {
  // expected-error @+1 {{'ks.batch_matmul' op operand element types must match}}
  %0 = ks.batch_matmul %a, %b : tensor<2x64x128xf32>, tensor<2x128x256xf16> -> tensor<2x64x256xf32>
  return
}

// -----

func.func @batch_matmul_inner_dim_mismatch(%a: tensor<2x64x128xf32>, %b: tensor<2x64x256xf32>) {
  // expected-error @+1 {{'ks.batch_matmul' op inner dimensions must match: lhs has 128, rhs has 64}}
  %0 = ks.batch_matmul %a, %b : tensor<2x64x128xf32>, tensor<2x64x256xf32> -> tensor<2x64x256xf32>
  return
}

// -----

func.func @batch_matmul_batch_dim_mismatch(%a: tensor<2x64x128xf32>, %b: tensor<4x128x256xf32>) {
  // expected-error @+1 {{'ks.batch_matmul' op batch dimension 0 must match: lhs has 2, rhs has 4}}
  %0 = ks.batch_matmul %a, %b : tensor<2x64x128xf32>, tensor<4x128x256xf32> -> tensor<2x64x256xf32>
  return
}

// -----

func.func @batch_matmul_rank_mismatch(%a: tensor<2x64x128xf32>, %b: tensor<3x2x128x256xf32>) {
  // expected-error @+1 {{'ks.batch_matmul' op operands must have the same rank}}
  %0 = ks.batch_matmul %a, %b : tensor<2x64x128xf32>, tensor<3x2x128x256xf32> -> tensor<2x64x256xf32>
  return
}
