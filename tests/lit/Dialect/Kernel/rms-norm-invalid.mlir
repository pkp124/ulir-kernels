// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @rms_norm_unranked(%arg0: tensor<*xf32>, %w: tensor<64xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{input and weight must be ranked tensors}}
  %0 = ks.rms_norm %arg0, %w : tensor<*xf32>, tensor<64xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @rms_norm_unranked_weight(%arg0: tensor<32x64xf32>, %w: tensor<*xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{input and weight must be ranked tensors}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xf32>, tensor<*xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}
