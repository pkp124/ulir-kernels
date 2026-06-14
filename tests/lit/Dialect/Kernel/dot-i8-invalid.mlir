// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @dot_i8_unranked_input(%arg0: tensor<*xi8>,
                                 %arg1: tensor<7xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op operands and result must be ranked tensors}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<*xi8>, tensor<7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_ranked_2d_input(%arg0: tensor<1x7xi8>,
                                  %arg1: tensor<7xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input must be a 1D tensor}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<1x7xi8>, tensor<7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_ranked_2d_weight(%arg0: tensor<7xi8>,
                                   %arg1: tensor<1x7xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op weight must be a 1D tensor}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi8>, tensor<1x7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_non_i8_input(%arg0: tensor<7xi32>,
                               %arg1: tensor<7xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input element type must be signless i8}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi32>, tensor<7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_unsigned_input(%arg0: tensor<7xui8>,
                                 %arg1: tensor<7xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input element type must be signless i8}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xui8>, tensor<7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_non_i8_weight(%arg0: tensor<7xi8>,
                                %arg1: tensor<7xi32>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op weight element type must be signless i8}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi8>, tensor<7xi32> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_unsigned_weight(%arg0: tensor<7xi8>,
                                  %arg1: tensor<7xui8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op weight element type must be signless i8}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi8>, tensor<7xui8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_result_rank(%arg0: tensor<7xi8>,
                              %arg1: tensor<7xi8>) -> tensor<1xi32> {
  // expected-error @+1 {{'ks.dot_i8' op result must be a rank-0 tensor}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi8>, tensor<7xi8> -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

func.func @dot_i8_result_type(%arg0: tensor<7xi8>,
                              %arg1: tensor<7xi8>) -> tensor<i64> {
  // expected-error @+1 {{'ks.dot_i8' op result element type must be i32}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi8>, tensor<7xi8> -> tensor<i64>
  return %0 : tensor<i64>
}

// -----

func.func @dot_i8_shape_mismatch(%arg0: tensor<7xi8>,
                                 %arg1: tensor<8xi8>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input and weight dimensions must match}}
  %0 = ks.dot_i8 %arg0, %arg1 : tensor<7xi8>, tensor<8xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_input_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                %arg1: tensor<7xi8>)
    -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op input_zero_point 128 does not fit in i8}}
  %0 = ks.dot_i8 %arg0, %arg1 {input_zero_point = 128 : i64}
      : tensor<7xi8>, tensor<7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_i8_weight_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                 %arg1: tensor<7xi8>)
    -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_i8' op weight_zero_point -129 does not fit in i8}}
  %0 = ks.dot_i8 %arg0, %arg1 {weight_zero_point = -129 : i64}
      : tensor<7xi8>, tensor<7xi8> -> tensor<i32>
  return %0 : tensor<i32>
}
