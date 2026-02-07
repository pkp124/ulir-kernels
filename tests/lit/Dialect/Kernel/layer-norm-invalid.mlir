// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @layer_norm_unranked(%input: tensor<*xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) {
  // expected-error @+1 {{'ks.layer_norm' op input must be a ranked tensor}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<*xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<*xf32>
  return
}
