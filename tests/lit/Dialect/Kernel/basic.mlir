// RUN: %ks-opt %s | %FileCheck %s

// Basic parsing and printing test for KernelSmith dialect

// CHECK-LABEL: func @test_relu
func.func @test_relu(%input: tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: ks.relu
  %output = ks.relu %input : tensor<32xf32>
  return %output : tensor<32xf32>
}

// CHECK-LABEL: func @test_gelu
func.func @test_gelu(%input: tensor<64xf32>) -> tensor<64xf32> {
  // CHECK: ks.gelu
  %output = ks.gelu %input : tensor<64xf32>
  return %output : tensor<64xf32>
}

// CHECK-LABEL: func @test_softmax
func.func @test_softmax(%input: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // CHECK: ks.softmax
  %output = ks.softmax %input : tensor<32x64xf32>
  return %output : tensor<32x64xf32>
}
