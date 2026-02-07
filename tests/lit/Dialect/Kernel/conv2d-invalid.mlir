// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @conv2d_input_not_4d(%input: tensor<32x32x3xf32>, %filter: tensor<3x3x3x16xf32>) {
  // expected-error @+1 {{'ks.conv2d' op input must be 4D tensor (NHWC)}}
  %0 = ks.conv2d %input, %filter : tensor<32x32x3xf32>, tensor<3x3x3x16xf32> -> tensor<1x30x30x16xf32>
  return
}

// -----

func.func @conv2d_filter_not_4d(%input: tensor<1x32x32x3xf32>, %filter: tensor<3x3x16xf32>) {
  // expected-error @+1 {{'ks.conv2d' op filter must be 4D tensor (HWIO)}}
  %0 = ks.conv2d %input, %filter : tensor<1x32x32x3xf32>, tensor<3x3x16xf32> -> tensor<1x30x30x16xf32>
  return
}
