// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @matvec_w4a8_unranked_input(%arg0: tensor<*xi8>,
                                      %arg1: tensor<3x4xi8>,
                                      %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op operands and result must be ranked tensors}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<*xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_input_rank(%arg0: tensor<1x7xi8>,
                                  %arg1: tensor<3x4xi8>,
                                  %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op input must be a 1D tensor}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<1x7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_packed_weights_rank(%arg0: tensor<7xi8>,
                                           %arg1: tensor<4xi8>,
                                           %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op packed_weights must be a 2D tensor}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_weight_scales_rank(%arg0: tensor<7xi8>,
                                          %arg1: tensor<3x4xi8>,
                                          %arg2: tensor<2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op weight_scales must be a 2D tensor}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_non_i8_input(%arg0: tensor<7xi32>,
                                    %arg1: tensor<3x4xi8>,
                                    %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op input element type must be signless i8}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi32>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_non_i8_packed_weights(%arg0: tensor<7xi8>,
                                             %arg1: tensor<3x4xf32>,
                                             %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op packed_weights element type must be signless i8}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xf32>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_non_f32_weight_scales(%arg0: tensor<7xi8>,
                                             %arg1: tensor<3x4xi8>,
                                             %arg2: tensor<3x2xf64>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op weight_scales element type must be f32}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf64> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_result_rank(%arg0: tensor<7xi8>,
                                   %arg1: tensor<3x4xi8>,
                                   %arg2: tensor<3x2xf32>)
    -> tensor<1x3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op result must be a 1D tensor}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// -----

func.func @matvec_w4a8_result_type(%arg0: tensor<7xi8>,
                                   %arg1: tensor<3x4xi8>,
                                   %arg2: tensor<3x2xf32>)
    -> tensor<3xf64> {
  // expected-error @+1 {{'ks.matvec_w4a8' op result element type must be f32}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

// -----

func.func @matvec_w4a8_packed_col_mismatch(%arg0: tensor<7xi8>,
                                           %arg1: tensor<3x3xi8>,
                                           %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op packed_weights column dimension must equal ceil(input dimension / 2)}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x3xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_scale_col_mismatch(%arg0: tensor<7xi8>,
                                          %arg1: tensor<3x4xi8>,
                                          %arg2: tensor<3x1xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op weight_scales column dimension must equal ceil(input dimension / group_size)}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x1xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_scale_row_mismatch(%arg0: tensor<7xi8>,
                                          %arg1: tensor<3x4xi8>,
                                          %arg2: tensor<4x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op weight_scales row dimension must match packed_weights row dimension}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<4x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_result_row_mismatch(%arg0: tensor<7xi8>,
                                           %arg1: tensor<3x4xi8>,
                                           %arg2: tensor<3x2xf32>)
    -> tensor<4xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op result dimension must match packed_weights row dimension}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @matvec_w4a8_non_positive_group_size(%arg0: tensor<7xi8>,
                                               %arg1: tensor<3x4xi8>,
                                               %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op group_size must be positive}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 0 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_non_positive_input_scale(%arg0: tensor<7xi8>,
                                                %arg1: tensor<3x4xi8>,
                                                %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op input_scale must be positive and finite}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 0.000000e+00 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_input_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                     %arg1: tensor<3x4xi8>,
                                                     %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op input_zero_point 128 does not fit in i8}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64,
       input_zero_point = 128 : i64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @matvec_w4a8_weight_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                      %arg1: tensor<3x4xi8>,
                                                      %arg2: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // expected-error @+1 {{'ks.matvec_w4a8' op weight_zero_point -9 does not fit in i4}}
  %0 = ks.matvec_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64,
       weight_zero_point = -9 : i64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
