// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @matvec_i8_unranked_input(%arg0: tensor<*xi8>,
                                    %arg1: tensor<3x7xi8>)
    -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op operands and result must be ranked tensors}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<*xi8>, tensor<3x7xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_input_rank(%arg0: tensor<1x7xi8>,
                                %arg1: tensor<3x7xi8>) -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op input must be a 1D tensor}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<1x7xi8>, tensor<3x7xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_weight_rank(%arg0: tensor<7xi8>,
                                 %arg1: tensor<7xi8>) -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op weights must be a 2D tensor}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi8>, tensor<7xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_non_i8_input(%arg0: tensor<7xi32>,
                                  %arg1: tensor<3x7xi8>) -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op input element type must be i8}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi32>, tensor<3x7xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_non_i8_weights(%arg0: tensor<7xi8>,
                                    %arg1: tensor<3x7xi32>) -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op weights element type must be i8}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi8>, tensor<3x7xi32> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_result_rank(%arg0: tensor<7xi8>,
                                 %arg1: tensor<3x7xi8>) -> tensor<1x3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op result must be a 1D tensor}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

func.func @matvec_i8_result_type(%arg0: tensor<7xi8>,
                                 %arg1: tensor<3x7xi8>) -> tensor<3xi64> {
  // expected-error @+1 {{'ks.matvec_i8' op result element type must be i32}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// -----

func.func @matvec_i8_col_mismatch(%arg0: tensor<7xi8>,
                                  %arg1: tensor<3x8xi8>) -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op input dimension must match weights column dimension}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi8>, tensor<3x8xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_row_mismatch(%arg0: tensor<7xi8>,
                                  %arg1: tensor<3x7xi8>) -> tensor<4xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op result dimension must match weights row dimension}}
  %0 = ks.matvec_i8 %arg0, %arg1
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

func.func @matvec_i8_input_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                   %arg1: tensor<3x7xi8>)
    -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op input_zero_point 128 does not fit in i8}}
  %0 = ks.matvec_i8 %arg0, %arg1 {input_zero_point = 128 : i64}
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func.func @matvec_i8_weight_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                    %arg1: tensor<3x7xi8>)
    -> tensor<3xi32> {
  // expected-error @+1 {{'ks.matvec_i8' op weight_zero_point -129 does not fit in i8}}
  %0 = ks.matvec_i8 %arg0, %arg1 {weight_zero_point = -129 : i64}
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<3xi32>
  return %0 : tensor<3xi32>
}
