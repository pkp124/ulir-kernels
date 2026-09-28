// f32 matmul: C[64,256] = A[64,128] * B[128,256].
//
//   build/bin/ks-opt examples/matmul.mlir --ks-lower-to-linalg --ks-tile
//   ./scripts/compile-rvv.sh examples/matmul.mlir /tmp/matmul.s

func.func @gemm(%A: tensor<64x128xf32>,
                %B: tensor<128x256xf32>) -> tensor<64x256xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32>
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}
