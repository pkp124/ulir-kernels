// End-to-end test: 256x256 f32 matrix multiplication

module {
  func.func @matmul_256x256(%A: tensor<256x256xf32>, %B: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %C = ks.matmul %A, %B : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    return %C : tensor<256x256xf32>
  }
}
