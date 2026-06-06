// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @quantize_unranked_input(%arg0: tensor<*xf32>) -> tensor<16xi8> {
  // expected-error @+1 {{'ks.quantize' op input and result must be ranked tensors}}
  %0 = ks.quantize %arg0 {scale = 1.000000e+00 : f64} : tensor<*xf32> -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

func.func @quantize_non_float_input(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  // expected-error @+1 {{'ks.quantize' op input element type must be floating-point}}
  %0 = ks.quantize %arg0 {scale = 1.000000e+00 : f64} : tensor<16xi8> -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

func.func @quantize_non_integer_result(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error @+1 {{'ks.quantize' op result element type must be integer}}
  %0 = ks.quantize %arg0 {scale = 1.000000e+00 : f64} : tensor<16xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @quantize_shape_mismatch(%arg0: tensor<16xf32>) -> tensor<8xi8> {
  // expected-error @+1 {{'ks.quantize' op input and result shapes must match}}
  %0 = ks.quantize %arg0 {scale = 1.000000e+00 : f64} : tensor<16xf32> -> tensor<8xi8>
  return %0 : tensor<8xi8>
}

// -----

func.func @quantize_non_positive_scale(%arg0: tensor<16xf32>) -> tensor<16xi8> {
  // expected-error @+1 {{'ks.quantize' op scale must be positive and finite}}
  %0 = ks.quantize %arg0 {scale = 0.000000e+00 : f64} : tensor<16xf32> -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

func.func @quantize_zero_point_out_of_range(%arg0: tensor<16xf32>) -> tensor<16xi8> {
  // expected-error @+1 {{'ks.quantize' op zero_point 128 does not fit in integer element type 'i8'}}
  %0 = ks.quantize %arg0 {scale = 1.000000e+00 : f64, zero_point = 128 : i64} : tensor<16xf32> -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

func.func @dequantize_unranked_input(%arg0: tensor<*xi8>) -> tensor<16xf32> {
  // expected-error @+1 {{'ks.dequantize' op input and result must be ranked tensors}}
  %0 = ks.dequantize %arg0 {scale = 1.000000e+00 : f64} : tensor<*xi8> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @dequantize_non_integer_input(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error @+1 {{'ks.dequantize' op input element type must be integer}}
  %0 = ks.dequantize %arg0 {scale = 1.000000e+00 : f64} : tensor<16xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @dequantize_non_float_result(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  // expected-error @+1 {{'ks.dequantize' op result element type must be floating-point}}
  %0 = ks.dequantize %arg0 {scale = 1.000000e+00 : f64} : tensor<16xi8> -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

func.func @dequantize_shape_mismatch(%arg0: tensor<16xi8>) -> tensor<8xf32> {
  // expected-error @+1 {{'ks.dequantize' op input and result shapes must match}}
  %0 = ks.dequantize %arg0 {scale = 1.000000e+00 : f64} : tensor<16xi8> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @dequantize_non_positive_scale(%arg0: tensor<16xi8>) -> tensor<16xf32> {
  // expected-error @+1 {{'ks.dequantize' op scale must be positive and finite}}
  %0 = ks.dequantize %arg0 {scale = -1.000000e+00 : f64} : tensor<16xi8> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

func.func @dequantize_zero_point_out_of_range(%arg0: tensor<16xi8>) -> tensor<16xf32> {
  // expected-error @+1 {{'ks.dequantize' op zero_point -129 does not fit in integer element type 'i8'}}
  %0 = ks.dequantize %arg0 {scale = 1.000000e+00 : f64, zero_point = -129 : i64} : tensor<16xi8> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
