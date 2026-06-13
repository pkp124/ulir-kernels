// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_dot_i8
func.func @test_dot_i8(%input: tensor<16xi8>,
                       %weight: tensor<16xi8>) -> tensor<i32> {
  // CHECK: ks.dot_i8
  %out = ks.dot_i8 %input, %weight {input_zero_point = -2 : i64, weight_zero_point = 3 : i64} : tensor<16xi8>, tensor<16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// CHECK-LABEL: func @test_matvec_i8
func.func @test_matvec_i8(%input: tensor<16xi8>,
                          %weights: tensor<4x16xi8>) -> tensor<4xi32> {
  // CHECK: ks.matvec_i8
  %out = ks.matvec_i8 %input, %weights {input_zero_point = 1 : i64, weight_zero_point = -3 : i64} : tensor<16xi8>, tensor<4x16xi8> -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// CHECK-LABEL: func @test_matvec_i8_dynamic_rows
func.func @test_matvec_i8_dynamic_rows(%input: tensor<16xi8>,
                                       %weights: tensor<?x16xi8>) -> tensor<?xi32> {
  // CHECK: ks.matvec_i8
  %out = ks.matvec_i8 %input, %weights : tensor<16xi8>, tensor<?x16xi8> -> tensor<?xi32>
  return %out : tensor<?xi32>
}
