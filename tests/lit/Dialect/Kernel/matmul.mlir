// RUN: %ks-opt %s | %FileCheck %s

// Tests for ks.matmul operation

// CHECK-LABEL: func @test_matmul_basic
func.func @test_matmul_basic(%A: tensor<64x128xf32>, %B: tensor<128x256xf32>) 
    -> tensor<64x256xf32> {
  // CHECK: ks.matmul
  // CHECK-SAME: tensor<64x128xf32>, tensor<128x256xf32> -> tensor<64x256xf32>
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}

// CHECK-LABEL: func @test_matmul_square
func.func @test_matmul_square(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) 
    -> tensor<64x64xf32> {
  // CHECK: ks.matmul
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> 
                          -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: func @test_matmul_f16
func.func @test_matmul_f16(%A: tensor<32x64xf16>, %B: tensor<64x128xf16>)
    -> tensor<32x128xf16> {
  // CHECK: ks.matmul
  %C = ks.matmul %A, %B : tensor<32x64xf16>, tensor<64x128xf16>
                          -> tensor<32x128xf16>
  return %C : tensor<32x128xf16>
}

// CHECK-LABEL: func @test_matmul_acc_type
func.func @test_matmul_acc_type(%A: tensor<32x64xf16>, %B: tensor<64x128xf16>)
    -> tensor<32x128xf16> {
  // CHECK: ks.matmul
  // CHECK-SAME: acc_type = f32
  %C = ks.matmul %A, %B {acc_type = f32} : tensor<32x64xf16>, tensor<64x128xf16>
                                           -> tensor<32x128xf16>
  return %C : tensor<32x128xf16>
}
