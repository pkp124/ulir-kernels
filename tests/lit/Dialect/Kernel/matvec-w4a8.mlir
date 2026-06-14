// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_matvec_w4a8
func.func @test_matvec_w4a8(%input: tensor<7xi8>,
                            %packed_weights: tensor<3x4xi8>,
                            %weight_scales: tensor<3x2xf32>)
    -> tensor<3xf32> {
  // CHECK: ks.matvec_w4a8
  %output = ks.matvec_w4a8 %input, %packed_weights, %weight_scales
      {group_size = 4 : i64, input_scale = 5.000000e-01 : f64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x2xf32> -> tensor<3xf32>
  return %output : tensor<3xf32>
}

// CHECK-LABEL: func @test_matvec_w4a8_zero_points
func.func @test_matvec_w4a8_zero_points(%input: tensor<?xi8>,
                                        %packed_weights: tensor<?x?xi8>,
                                        %weight_scales: tensor<?x?xf32>)
    -> tensor<?xf32> {
  // CHECK: ks.matvec_w4a8
  %output = ks.matvec_w4a8 %input, %packed_weights, %weight_scales
      {group_size = 64 : i64, input_scale = 2.500000e-01 : f64,
       input_zero_point = -3 : i64, weight_zero_point = 2 : i64}
      : tensor<?xi8>, tensor<?x?xi8>, tensor<?x?xf32> -> tensor<?xf32>
  return %output : tensor<?xf32>
}
