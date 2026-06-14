// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_dot_w4a8
func.func @test_dot_w4a8(%input: tensor<7xi8>,
                         %packed_weight: tensor<4xui8>,
                         %weight_scales: tensor<2xf32>) -> tensor<f32> {
  // CHECK: ks.dot_w4a8
  %acc = ks.dot_w4a8 %input, %packed_weight, %weight_scales
      {group_size = 4 : i64, input_scale = 5.000000e-01 : f64}
      : tensor<7xi8>, tensor<4xui8>, tensor<2xf32> -> tensor<f32>
  return %acc : tensor<f32>
}

// CHECK-LABEL: func @test_dot_w4a8_zero_points
func.func @test_dot_w4a8_zero_points(%input: tensor<?xi8>,
                                     %packed_weight: tensor<?xui8>,
                                     %weight_scales: tensor<?xf32>)
    -> tensor<f32> {
  // CHECK: ks.dot_w4a8
  %acc = ks.dot_w4a8 %input, %packed_weight, %weight_scales
      {group_size = 64 : i64, input_scale = 2.500000e-01 : f64,
       input_zero_point = -3 : i64, weight_zero_point = 2 : i64}
      : tensor<?xi8>, tensor<?xui8>, tensor<?xf32> -> tensor<f32>
  return %acc : tensor<f32>
}
