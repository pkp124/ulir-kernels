// End-to-end test: 64x64 f32 matrix multiplication
// This file is compiled through the full pipeline and executed on QEMU

module {
  func.func @matmul_64x64(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
    return %C : tensor<64x64xf32>
  }
}
