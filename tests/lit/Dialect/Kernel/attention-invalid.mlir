// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @attention_query_not_3d(%q: tensor<64x32xf32>, %k: tensor<2x64x32xf32>, %v: tensor<2x64x32xf32>) {
  // expected-error @+1 {{'ks.attention' op query must be 3D tensor (batch, seq, dim)}}
  %0 = ks.attention %q, %k, %v : tensor<64x32xf32>, tensor<2x64x32xf32>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return
}

// -----

func.func @attention_key_not_3d(%q: tensor<2x64x32xf32>, %k: tensor<64x32xf32>, %v: tensor<2x64x32xf32>) {
  // expected-error @+1 {{'ks.attention' op key must be 3D tensor}}
  %0 = ks.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<64x32xf32>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return
}

// -----

func.func @attention_value_not_3d(%q: tensor<2x64x32xf32>, %k: tensor<2x64x32xf32>, %v: tensor<64x32xf32>) {
  // expected-error @+1 {{'ks.attention' op value must be 3D tensor}}
  %0 = ks.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x32xf32>, tensor<64x32xf32> -> tensor<2x64x32xf32>
  return
}
