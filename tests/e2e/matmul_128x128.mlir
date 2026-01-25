// End-to-end test: 128x128 f32 matrix multiplication

module {
  func.func @matmul_128x128(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %C = ks.matmul %A, %B : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
    return %C : tensor<128x128xf32>
  }
}
