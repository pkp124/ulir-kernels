// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_quantize
func.func @test_quantize(%input: tensor<16xf32>) -> tensor<16xi8> {
  // CHECK: ks.quantize
  %output = ks.quantize %input {scale = 5.000000e-01 : f64} : tensor<16xf32> -> tensor<16xi8>
  return %output : tensor<16xi8>
}

// CHECK-LABEL: func @test_quantize_zero_point
func.func @test_quantize_zero_point(%input: tensor<4x8xf32>) -> tensor<4x8xi8> {
  // CHECK: ks.quantize
  %output = ks.quantize %input {scale = 2.500000e-01 : f64, zero_point = -3 : i64} : tensor<4x8xf32> -> tensor<4x8xi8>
  return %output : tensor<4x8xi8>
}

// CHECK-LABEL: func @test_dequantize
func.func @test_dequantize(%input: tensor<16xi8>) -> tensor<16xf32> {
  // CHECK: ks.dequantize
  %output = ks.dequantize %input {scale = 5.000000e-01 : f64} : tensor<16xi8> -> tensor<16xf32>
  return %output : tensor<16xf32>
}

// CHECK-LABEL: func @test_dequantize_zero_point
func.func @test_dequantize_zero_point(%input: tensor<4x8xi8>) -> tensor<4x8xf32> {
  // CHECK: ks.dequantize
  %output = ks.dequantize %input {scale = 2.500000e-01 : f64, zero_point = -3 : i64} : tensor<4x8xi8> -> tensor<4x8xf32>
  return %output : tensor<4x8xf32>
}
