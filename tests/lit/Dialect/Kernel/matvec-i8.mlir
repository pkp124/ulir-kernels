// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_matvec_i8
func.func @test_matvec_i8(%input: tensor<7xi8>, %weights: tensor<3x7xi8>)
    -> tensor<3xi32> {
  // CHECK: ks.matvec_i8
  %output = ks.matvec_i8 %input, %weights
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<3xi32>
  return %output : tensor<3xi32>
}

// CHECK-LABEL: func @test_matvec_i8_zero_points
func.func @test_matvec_i8_zero_points(%input: tensor<?xi8>,
                                      %weights: tensor<?x?xi8>)
    -> tensor<?xi32> {
  // CHECK: ks.matvec_i8
  %output = ks.matvec_i8 %input, %weights
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<?xi8>, tensor<?x?xi8> -> tensor<?xi32>
  return %output : tensor<?xi32>
}
