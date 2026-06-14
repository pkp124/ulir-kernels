// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func @test_dot_i8
// CHECK-NOT:   ks.dot_i8
// CHECK-DAG:   %[[WEIGHT_ZP:.*]] = arith.constant 5 : i32
// CHECK-DAG:   %[[INPUT_ZP:.*]] = arith.constant -3 : i32
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<i32>
// CHECK:       %[[INIT:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[EMPTY]] : tensor<i32>) -> tensor<i32>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["reduction"]
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<7xi8>, tensor<7xi8>)
// CHECK-SAME:    outs(%[[INIT]] : tensor<i32>)
// CHECK:       ^bb0(%[[INPUT:.*]]: i8, %[[WEIGHT:.*]]: i8, %[[ACC:.*]]: i32):
// CHECK:         %[[INPUT_I32:.*]] = arith.extsi %[[INPUT]] : i8 to i32
// CHECK:         %[[INPUT_CENTERED:.*]] = arith.subi %[[INPUT_I32]], %[[INPUT_ZP]] : i32
// CHECK:         %[[WEIGHT_I32:.*]] = arith.extsi %[[WEIGHT]] : i8 to i32
// CHECK:         %[[WEIGHT_CENTERED:.*]] = arith.subi %[[WEIGHT_I32]], %[[WEIGHT_ZP]] : i32
// CHECK:         %[[PRODUCT:.*]] = arith.muli %[[INPUT_CENTERED]], %[[WEIGHT_CENTERED]] : i32
// CHECK:         %[[SUM:.*]] = arith.addi %[[ACC]], %[[PRODUCT]] : i32
// CHECK:         linalg.yield %[[SUM]] : i32
func.func @test_dot_i8(%input: tensor<7xi8>, %weight: tensor<7xi8>)
    -> tensor<i32> {
  %acc = ks.dot_i8 %input, %weight
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<7xi8>, tensor<7xi8> -> tensor<i32>
  return %acc : tensor<i32>
}

// CHECK-LABEL: func @test_dot_i8_dynamic
// CHECK-NOT:   ks.dot_i8
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<?xi8>, tensor<?xi8>)
func.func @test_dot_i8_dynamic(%input: tensor<?xi8>, %weight: tensor<?xi8>)
    -> tensor<i32> {
  %acc = ks.dot_i8 %input, %weight
      : tensor<?xi8>, tensor<?xi8> -> tensor<i32>
  return %acc : tensor<i32>
}
