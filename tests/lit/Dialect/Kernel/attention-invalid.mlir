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

// -----

func.func @attention_type_mismatch(%q: tensor<2x64x32xf32>, %k: tensor<2x64x32xf16>, %v: tensor<2x64x32xf32>) {
  // expected-error @+1 {{'ks.attention' op query, key, and value element types must match}}
  %0 = ks.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x32xf16>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return
}

// -----

func.func @attention_head_dim_mismatch(%q: tensor<2x64x32xf32>, %k: tensor<2x64x48xf32>, %v: tensor<2x64x32xf32>) {
  // expected-error @+1 {{'ks.attention' op query head dimension (32) must match key head dimension (48)}}
  %0 = ks.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x48xf32>, tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  return
}

// -----

func.func @attention_kv_seq_mismatch(%q: tensor<2x64x32xf32>, %k: tensor<2x64x32xf32>, %v: tensor<2x128x32xf32>) {
  // expected-error @+1 {{'ks.attention' op key sequence length (64) must match value sequence length (128)}}
  %0 = ks.attention %q, %k, %v : tensor<2x64x32xf32>, tensor<2x64x32xf32>, tensor<2x128x32xf32> -> tensor<2x64x32xf32>
  return
}
