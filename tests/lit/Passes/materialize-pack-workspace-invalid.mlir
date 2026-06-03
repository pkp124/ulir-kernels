// RUN: %ks-opt %s --ks-materialize-pack-workspace -split-input-file -verify-diagnostics

func.func @missing_workspace(%B: tensor<256x128xf32>) -> tensor<4x256x32xf32> {
  %empty = tensor.empty() : tensor<4x256x32xf32>
  // expected-error @+1 {{workspace argument required for materializing packed buffers}}
  %packed = linalg.pack %B outer_dims_perm = [1, 0]
      inner_dims_pos = [1] inner_tiles = [32]
      into %empty : tensor<256x128xf32> -> tensor<4x256x32xf32>
  return %packed : tensor<4x256x32xf32>
}

// -----

func.func @unsupported_layout(
    %B: tensor<256x128xf32>,
    %workspace: memref<?xi8>) -> tensor<32x128x8xf32> {
  %empty = tensor.empty() : tensor<32x128x8xf32>
  // expected-error @+1 {{unsupported linalg.pack layout for KernelSmith matmul}}
  %packed = linalg.pack %B outer_dims_perm = [0, 1]
      inner_dims_pos = [0] inner_tiles = [8]
      into %empty : tensor<256x128xf32> -> tensor<32x128x8xf32>
  return %packed : tensor<32x128x8xf32>
}

// -----

func.func @dynamic_shape(
    %B: tensor<256x?xf32>,
    %workspace: memref<?xi8>,
    %num_panels: index) -> tensor<?x256x32xf32> {
  %empty = tensor.empty(%num_panels) : tensor<?x256x32xf32>
  // expected-error @+1 {{dynamic pack workspace materialization is not supported}}
  %packed = linalg.pack %B outer_dims_perm = [1, 0]
      inner_dims_pos = [1] inner_tiles = [32]
      into %empty : tensor<256x?xf32> -> tensor<?x256x32xf32>
  return %packed : tensor<?x256x32xf32>
}
