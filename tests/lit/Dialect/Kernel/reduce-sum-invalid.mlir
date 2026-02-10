// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @reduce_sum_axis_out_of_range(%arg0: tensor<32x64xf32>) -> tensor<32xf32> {
  // expected-error @+1 {{axis 2 is out of range for tensor of rank 2}}
  %0 = ks.reduce_sum %arg0 {axes = [2]} : tensor<32x64xf32> -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

func.func @reduce_sum_axis_too_negative(%arg0: tensor<32x64xf32>) -> tensor<64xf32> {
  // expected-error @+1 {{axis -3 is out of range for tensor of rank 2}}
  %0 = ks.reduce_sum %arg0 {axes = [-3]} : tensor<32x64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

func.func @reduce_sum_duplicate_axes(%arg0: tensor<32x64xf32>) -> tensor<f32> {
  // expected-error @+1 {{duplicate axis 1}}
  %0 = ks.reduce_sum %arg0 {axes = [1, 1]} : tensor<32x64xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @reduce_sum_duplicate_axes_normalized(%arg0: tensor<32x64xf32>) -> tensor<f32> {
  // expected-error @+1 {{duplicate axis -1}}
  %0 = ks.reduce_sum %arg0 {axes = [1, -1]} : tensor<32x64xf32> -> tensor<f32>
  return %0 : tensor<f32>
}
