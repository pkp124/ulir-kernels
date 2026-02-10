// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @softmax_axis_too_large(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{axis 2 is out of range for tensor of rank 2}}
  %0 = ks.softmax %arg0 {axis = 2 : i64} : tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @softmax_axis_too_negative(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{axis -3 is out of range for tensor of rank 2}}
  %0 = ks.softmax %arg0 {axis = -3 : i64} : tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @softmax_axis_out_of_range_3d(%arg0: tensor<2x32x64xf32>) -> tensor<2x32x64xf32> {
  // expected-error @+1 {{axis 3 is out of range for tensor of rank 3}}
  %0 = ks.softmax %arg0 {axis = 3 : i64} : tensor<2x32x64xf32>
  return %0 : tensor<2x32x64xf32>
}
