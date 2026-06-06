// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @layer_norm_unranked(%input: tensor<*xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) {
  // expected-error @+1 {{'ks.layer_norm' op input must be a ranked tensor}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<*xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<*xf32>
  return
}

// -----

func.func @layer_norm_unranked_weight(%input: tensor<32x64xf32>, %weight: tensor<*xf32>, %bias: tensor<64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op weight must be a ranked tensor}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<*xf32>, tensor<64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_unranked_bias(%input: tensor<32x64xf32>, %weight: tensor<64xf32>, %bias: tensor<*xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op bias must be a ranked tensor}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<64xf32>, tensor<*xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_unranked_output(%input: tensor<32x64xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{'ks.layer_norm' op output must be a ranked tensor}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @layer_norm_integer_input(%input: tensor<32x64xi32>, %weight: tensor<64xi32>, %bias: tensor<64xi32>) -> tensor<32x64xi32> {
  // expected-error @+1 {{'ks.layer_norm' op input element type must be floating-point}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xi32>, tensor<64xi32>, tensor<64xi32> -> tensor<32x64xi32>
  return %0 : tensor<32x64xi32>
}

// -----

func.func @layer_norm_element_type_mismatch(%input: tensor<32x64xf32>, %weight: tensor<64xf64>, %bias: tensor<64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op input, weight, bias, and output element types must match}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<64xf64>, tensor<64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_empty_normalized_shape(%input: tensor<32x64xf32>, %weight: tensor<f32>, %bias: tensor<f32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op normalized_shape must not be empty}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = []} : tensor<32x64xf32>, tensor<f32>, tensor<f32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_normalized_shape_too_large(%input: tensor<32x64xf32>, %weight: tensor<2x32x64xf32>, %bias: tensor<2x32x64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op normalized_shape rank 3 exceeds input rank 2}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [2, 32, 64]} : tensor<32x64xf32>, tensor<2x32x64xf32>, tensor<2x32x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_nonpositive_normalized_dim(%input: tensor<32x64xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op normalized_shape dimensions must be positive}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [0]} : tensor<32x64xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_weight_shape_mismatch(%input: tensor<32x64xf32>, %weight: tensor<32xf32>, %bias: tensor<64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op weight shape must match normalized_shape}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<32xf32>, tensor<64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_bias_shape_mismatch(%input: tensor<32x64xf32>, %weight: tensor<64xf32>, %bias: tensor<32xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op bias shape must match normalized_shape}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<64xf32>, tensor<32xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

func.func @layer_norm_input_shape_mismatch(%input: tensor<32x32xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<32x32xf32> {
  // expected-error @+1 {{'ks.layer_norm' op input trailing dimensions must match normalized_shape}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x32xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

func.func @layer_norm_output_shape_mismatch(%input: tensor<32x64xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<32x32xf32> {
  // expected-error @+1 {{'ks.layer_norm' op output shape must match input shape}}
  %0 = ks.layer_norm %input, %weight, %bias {normalized_shape = [64]} : tensor<32x64xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

func.func @layer_norm_nonpositive_eps(%input: tensor<32x64xf32>, %weight: tensor<64xf32>, %bias: tensor<64xf32>) -> tensor<32x64xf32> {
  // expected-error @+1 {{'ks.layer_norm' op eps must be positive and finite}}
  %0 = ks.layer_norm %input, %weight, %bias {eps = 0.000000e+00 : f32, normalized_shape = [64]} : tensor<32x64xf32>, tensor<64xf32>, tensor<64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}
