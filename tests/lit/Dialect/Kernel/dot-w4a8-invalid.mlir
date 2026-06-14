// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @dot_w4a8_unranked_input(%arg0: tensor<*xi8>,
                                   %arg1: tensor<4xui8>,
                                   %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op operands and result must be ranked tensors}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<*xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_input_rank(%arg0: tensor<1x7xi8>,
                               %arg1: tensor<4xui8>,
                               %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op input must be a 1D tensor}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<1x7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_packed_weight_rank(%arg0: tensor<7xi8>,
                                       %arg1: tensor<1x4xui8>,
                                       %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op packed_weight must be a 1D tensor}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<1x4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_weight_scales_rank(%arg0: tensor<7xi8>,
                                       %arg1: tensor<4xui8>,
                                       %arg2: tensor<1x2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op weight_scales must be a 1D tensor}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<1x2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_non_i8_input(%arg0: tensor<7xi32>,
                                 %arg1: tensor<4xui8>,
                                 %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op input element type must be i8}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi32>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_non_i8_packed_weight(%arg0: tensor<7xi8>,
                                         %arg1: tensor<4xf32>,
                                         %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op packed_weight element type must be 8-bit integer}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xf32>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_non_f32_weight_scales(%arg0: tensor<7xi8>,
                                          %arg1: tensor<4xui8>,
                                          %arg2: tensor<2xf64>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op weight_scales element type must be f32}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf64> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_result_rank(%arg0: tensor<7xi8>,
                                %arg1: tensor<4xui8>,
                                %arg2: tensor<2xf32>) -> tensor<1xf32> {
  // expected-error @+1 {{'ks.dot_w4a8' op result must be a rank-0 tensor}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func.func @dot_w4a8_result_type(%arg0: tensor<7xi8>,
                                %arg1: tensor<4xui8>,
                                %arg2: tensor<2xf32>) -> tensor<i32> {
  // expected-error @+1 {{'ks.dot_w4a8' op result element type must be f32}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @dot_w4a8_packed_weight_mismatch(%arg0: tensor<7xi8>,
                                           %arg1: tensor<3xui8>,
                                           %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op packed_weight dimension must equal ceil(input dimension / 2)}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<3xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_weight_scales_mismatch(%arg0: tensor<7xi8>,
                                           %arg1: tensor<4xui8>,
                                           %arg2: tensor<1xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op weight_scales dimension must equal ceil(input dimension / group_size)}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<1xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_non_positive_group_size(%arg0: tensor<7xi8>,
                                            %arg1: tensor<4xui8>,
                                            %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op group_size must be positive}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 0 : i64, input_scale = 1.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_non_positive_input_scale(%arg0: tensor<7xi8>,
                                             %arg1: tensor<4xui8>,
                                             %arg2: tensor<2xf32>) -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op input_scale must be positive and finite}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 0.000000e+00 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_input_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                  %arg1: tensor<4xui8>,
                                                  %arg2: tensor<2xf32>)
    -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op input_zero_point 128 does not fit in i8}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64,
       input_zero_point = 128 : i64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dot_w4a8_weight_zero_point_out_of_range(%arg0: tensor<7xi8>,
                                                   %arg1: tensor<4xui8>,
                                                   %arg2: tensor<2xf32>)
    -> tensor<f32> {
  // expected-error @+1 {{'ks.dot_w4a8' op weight_zero_point -9 does not fit in i4}}
  %0 = ks.dot_w4a8 %arg0, %arg1, %arg2
      {group_size = 4 : i64, input_scale = 1.000000e+00 : f64,
       weight_zero_point = -9 : i64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %0 : tensor<f32>
}
