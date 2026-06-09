// RUN: %ks-opt %s --ks-pack -split-input-file -verify-diagnostics

func.func @test_pack_rejects_non_divisible_n(
    %A: tensor<16x64xf32>,
    %B: tensor<64x33xf32>,
    %C: tensor<16x33xf32>) -> tensor<16x33xf32> {
  // expected-error @+1 {{requires rhs N dimension (33) to be divisible by pack-factor (32); tail handling is not supported by --ks-pack}}
  %D = linalg.matmul ins(%A, %B : tensor<16x64xf32>, tensor<64x33xf32>)
                     outs(%C : tensor<16x33xf32>) -> tensor<16x33xf32>
  return %D : tensor<16x33xf32>
}
