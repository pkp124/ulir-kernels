// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func @test_add_same_shape
// CHECK-NOT:   ks.add
// CHECK:       tensor.empty() : tensor<4x8xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP2D:.*]], #[[MAP2D]], #[[MAP2D]]]
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:       ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK:         %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
// CHECK:         linalg.yield %[[SUM]] : f32
func.func @test_add_same_shape(%lhs: tensor<4x8xf32>,
                               %rhs: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %out = ks.add %lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

// CHECK-LABEL: func @test_mul_rank_broadcast
// CHECK-NOT:   ks.mul
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP2D]], #[[MAP_TRAILING:.*]], #[[MAP2D]]]
// CHECK:       ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK:         %[[PRODUCT:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:         linalg.yield %[[PRODUCT]] : f32
func.func @test_mul_rank_broadcast(%lhs: tensor<2x4xf32>,
                                   %rhs: tensor<4xf32>) -> tensor<2x4xf32> {
  %out = ks.mul %lhs, %rhs : tensor<2x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// CHECK-LABEL: func @test_add_dim_one_broadcast
// CHECK-NOT:   ks.add
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP_DIM_ONE:.*]], #[[MAP2D]], #[[MAP2D]]]
// CHECK:       arith.addf
func.func @test_add_dim_one_broadcast(%lhs: tensor<2x1xf32>,
                                      %rhs: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %out = ks.add %lhs, %rhs : tensor<2x1xf32>, tensor<2x4xf32> -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// CHECK-LABEL: func @test_mul_dynamic
// CHECK-NOT:   ks.mul
// CHECK:       tensor.dim %{{.*}}, %c0
// CHECK:       tensor.empty(%{{.*}}) : tensor<?x4xf32>
// CHECK:       linalg.generic
// CHECK:       arith.mulf
func.func @test_mul_dynamic(%lhs: tensor<?x4xf32>,
                            %rhs: tensor<1x4xf32>) -> tensor<?x4xf32> {
  %out = ks.mul %lhs, %rhs : tensor<?x4xf32>, tensor<1x4xf32> -> tensor<?x4xf32>
  return %out : tensor<?x4xf32>
}
