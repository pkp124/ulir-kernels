// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_dot_i8
func.func @test_dot_i8(%input: tensor<7xi8>, %weight: tensor<7xi8>)
    -> tensor<i32> {
  // CHECK: ks.dot_i8
  %acc = ks.dot_i8 %input, %weight
      : tensor<7xi8>, tensor<7xi8> -> tensor<i32>
  return %acc : tensor<i32>
}

// CHECK-LABEL: func @test_dot_i8_zero_points
func.func @test_dot_i8_zero_points(%input: tensor<?xi8>,
                                   %weight: tensor<?xi8>) -> tensor<i32> {
  // CHECK: ks.dot_i8
  %acc = ks.dot_i8 %input, %weight
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<?xi8>, tensor<?xi8> -> tensor<i32>
  return %acc : tensor<i32>
}
