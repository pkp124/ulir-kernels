// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func @test_matvec_i8
// CHECK-NOT:   ks.matvec_i8
// CHECK-DAG:   %[[WEIGHT_ZP:.*]] = arith.constant 5 : i32
// CHECK-DAG:   %[[INPUT_ZP:.*]] = arith.constant -3 : i32
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<3xi32>
// CHECK:       %[[INIT:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[EMPTY]] : tensor<3xi32>) -> tensor<3xi32>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<7xi8>, tensor<3x7xi8>)
// CHECK-SAME:    outs(%[[INIT]] : tensor<3xi32>)
// CHECK:       ^bb0(%[[INPUT:.*]]: i8, %[[WEIGHT:.*]]: i8, %[[ACC:.*]]: i32):
// CHECK:         %[[INPUT_I32:.*]] = arith.extsi %[[INPUT]] : i8 to i32
// CHECK:         %[[INPUT_CENTERED:.*]] = arith.subi %[[INPUT_I32]], %[[INPUT_ZP]] : i32
// CHECK:         %[[WEIGHT_I32:.*]] = arith.extsi %[[WEIGHT]] : i8 to i32
// CHECK:         %[[WEIGHT_CENTERED:.*]] = arith.subi %[[WEIGHT_I32]], %[[WEIGHT_ZP]] : i32
// CHECK:         %[[PRODUCT:.*]] = arith.muli %[[INPUT_CENTERED]], %[[WEIGHT_CENTERED]] : i32
// CHECK:         %[[SUM:.*]] = arith.addi %[[ACC]], %[[PRODUCT]] : i32
// CHECK:         linalg.yield %[[SUM]] : i32
func.func @test_matvec_i8(%input: tensor<7xi8>, %weights: tensor<3x7xi8>)
    -> tensor<3xi32> {
  %output = ks.matvec_i8 %input, %weights
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<7xi8>, tensor<3x7xi8> -> tensor<3xi32>
  return %output : tensor<3xi32>
}

// CHECK-LABEL: func @test_matvec_i8_dynamic_rows
// CHECK-NOT:   ks.matvec_i8
// CHECK:       tensor.dim %{{.*}}, %c0
// CHECK:       tensor.empty(%{{.*}}) : tensor<?xi32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<7xi8>, tensor<?x7xi8>)
func.func @test_matvec_i8_dynamic_rows(%input: tensor<7xi8>,
                                       %weights: tensor<?x7xi8>)
    -> tensor<?xi32> {
  %output = ks.matvec_i8 %input, %weights
      : tensor<7xi8>, tensor<?x7xi8> -> tensor<?xi32>
  return %output : tensor<?xi32>
}
