// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @dot_i8_input_not_1d(%input: tensor<2x8xi8>, %weight: tensor<16xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input must be a 1D tensor}}
  %out = ks.dot_i8 %input, %weight : tensor<2x8xi8>, tensor<16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// -----

func.func @dot_i8_weight_not_1d(%input: tensor<16xi8>, %weight: tensor<1x16xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op weight must be a 1D tensor}}
  %out = ks.dot_i8 %input, %weight : tensor<16xi8>, tensor<1x16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// -----

func.func @dot_i8_result_not_scalar(%input: tensor<16xi8>, %weight: tensor<16xi8>) -> tensor<1xi32> {
  // expected-error @+1 {{'ks.dot_i8' op result must be a rank-0 i32 tensor}}
  %out = ks.dot_i8 %input, %weight : tensor<16xi8>, tensor<16xi8> -> tensor<1xi32>
  return %out : tensor<1xi32>
}

// -----

func.func @dot_i8_input_not_i8(%input: tensor<16xi16>, %weight: tensor<16xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input element type must be i8}}
  %out = ks.dot_i8 %input, %weight : tensor<16xi16>, tensor<16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// -----

func.func @dot_i8_dim_mismatch(%input: tensor<15xi8>, %weight: tensor<16xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input and weight dimensions must match}}
  %out = ks.dot_i8 %input, %weight : tensor<15xi8>, tensor<16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// -----

func.func @dot_i8_zero_point_out_of_range(%input: tensor<16xi8>, %weight: tensor<16xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op zero_point 128 does not fit in integer element type 'i8'}}
  %out = ks.dot_i8 %input, %weight {input_zero_point = 128 : i64} : tensor<16xi8>, tensor<16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// -----

func.func @matvec_i8_input_not_1d(%input: tensor<1x16xi8>, %weights: tensor<4x16xi8>) -> tensor<4xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op input must be a 1D tensor}}
  %out = ks.matvec_i8 %input, %weights : tensor<1x16xi8>, tensor<4x16xi8> -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @matvec_i8_weights_not_2d(%input: tensor<16xi8>, %weights: tensor<16xi8>) -> tensor<4xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op weights must be a 2D tensor}}
  %out = ks.matvec_i8 %input, %weights : tensor<16xi8>, tensor<16xi8> -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @matvec_i8_result_not_1d(%input: tensor<16xi8>, %weights: tensor<4x16xi8>) -> tensor<4x1xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op result must be a 1D i32 tensor}}
  %out = ks.matvec_i8 %input, %weights : tensor<16xi8>, tensor<4x16xi8> -> tensor<4x1xi32>
  return %out : tensor<4x1xi32>
}

// -----

func.func @matvec_i8_weights_not_i8(%input: tensor<16xi8>, %weights: tensor<4x16xi16>) -> tensor<4xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op weights element type must be i8}}
  %out = ks.matvec_i8 %input, %weights : tensor<16xi8>, tensor<4x16xi16> -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @matvec_i8_col_mismatch(%input: tensor<15xi8>, %weights: tensor<4x16xi8>) -> tensor<4xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op input dimension must match weights column dimension}}
  %out = ks.matvec_i8 %input, %weights : tensor<15xi8>, tensor<4x16xi8> -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @matvec_i8_row_mismatch(%input: tensor<16xi8>, %weights: tensor<4x16xi8>) -> tensor<5xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op result dimension must match weights row dimension}}
  %out = ks.matvec_i8 %input, %weights : tensor<16xi8>, tensor<4x16xi8> -> tensor<5xi32>
  return %out : tensor<5xi32>
}
