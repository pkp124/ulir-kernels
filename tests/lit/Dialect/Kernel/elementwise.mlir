// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL: func @test_add_same_shape
func.func @test_add_same_shape(%lhs: tensor<4x8xf32>,
                               %rhs: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: ks.add
  %out = ks.add %lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

// CHECK-LABEL: func @test_mul_broadcast_rhs
func.func @test_mul_broadcast_rhs(%lhs: tensor<2x4xf32>,
                                  %rhs: tensor<4xf32>) -> tensor<2x4xf32> {
  // CHECK: ks.mul
  %out = ks.mul %lhs, %rhs : tensor<2x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// CHECK-LABEL: func @test_add_broadcast_dim_one
func.func @test_add_broadcast_dim_one(%lhs: tensor<2x1xf32>,
                                      %rhs: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // CHECK: ks.add
  %out = ks.add %lhs, %rhs : tensor<2x1xf32>, tensor<2x4xf32> -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// CHECK-LABEL: func @test_mul_dynamic
func.func @test_mul_dynamic(%lhs: tensor<?x4xf32>,
                            %rhs: tensor<1x4xf32>) -> tensor<?x4xf32> {
  // CHECK: ks.mul
  %out = ks.mul %lhs, %rhs : tensor<?x4xf32>, tensor<1x4xf32> -> tensor<?x4xf32>
  return %out : tensor<?x4xf32>
}
