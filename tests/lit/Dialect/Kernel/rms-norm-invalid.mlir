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

// -----

func.func @rms_norm_unranked_output(%arg0: tensor<32x64xf32>, %w: tensor<64xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{output must be a ranked tensor}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xf32>, tensor<64xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @rms_norm_integer_input(%arg0: tensor<32x64xi32>, %w: tensor<64xi32>) -> tensor<32x64xi32> {
  // expected-error @+1 {{input element type must be floating-point}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xi32>, tensor<64xi32> -> tensor<32x64xi32>
  return %0 : tensor<32x64xi32>
}

// -----

func.func @rms_norm_element_type_mismatch(%arg0: tensor<32x64xf32>, %w: tensor<64xf64>) -> tensor<32x64xf32> {
  // expected-error @+1 {{input, weight, and output element types must match}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xf32>, tensor<64xf64> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @rms_norm_output_shape_mismatch(%arg0: tensor<32x64xf32>, %w: tensor<64xf32>) -> tensor<32x32xf32> {
  // expected-error @+1 {{output shape must match input shape}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xf32>, tensor<64xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

func.func @rms_norm_weight_rank_mismatch(%arg0: tensor<32x64xf32>, %w: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{weight must be a 1D tensor}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @rms_norm_weight_shape_mismatch(%arg0: tensor<32x64xf32>, %w: tensor<32xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{weight dimension must match input last dimension}}
  %0 = ks.rms_norm %arg0, %w : tensor<32x64xf32>, tensor<32xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @rms_norm_nonpositive_eps(%arg0: tensor<32x64xf32>, %w: tensor<64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{eps must be positive and finite}}
  %0 = ks.rms_norm %arg0, %w {eps = 0.000000e+00 : f32} : tensor<32x64xf32>, tensor<64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}
