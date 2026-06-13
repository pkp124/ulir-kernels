// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func @test_dot_i8_lowering
// CHECK-NOT: ks.dot_i8
// CHECK-DAG: %[[IZP:.*]] = arith.constant -2 : i32
// CHECK-DAG: %[[WZP:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: linalg.fill ins(%[[ZERO]]
// CHECK: linalg.generic
// CHECK: arith.extsi
// CHECK: arith.subi {{.*}}, %[[IZP]]
// CHECK: arith.subi {{.*}}, %[[WZP]]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: linalg.yield
func.func @test_dot_i8_lowering(%input: tensor<16xi8>,
                                %weight: tensor<16xi8>) -> tensor<i32> {
  %out = ks.dot_i8 %input, %weight {input_zero_point = -2 : i64, weight_zero_point = 3 : i64} : tensor<16xi8>, tensor<16xi8> -> tensor<i32>
  return %out : tensor<i32>
}

// CHECK-LABEL: func @test_matvec_i8_lowering
// CHECK-NOT: ks.matvec_i8
// CHECK-DAG: arith.constant 1 : i32
// CHECK-DAG: arith.constant -3 : i32
// CHECK-DAG: arith.constant 0 : i32
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.extsi
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: linalg.yield
func.func @test_matvec_i8_lowering(%input: tensor<16xi8>,
                                   %weights: tensor<4x16xi8>) -> tensor<4xi32> {
  %out = ks.matvec_i8 %input, %weights {input_zero_point = 1 : i64, weight_zero_point = -3 : i64} : tensor<16xi8>, tensor<4x16xi8> -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// CHECK-LABEL: func @test_matvec_i8_dynamic_rows
// CHECK-NOT: ks.matvec_i8
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.generic
func.func @test_matvec_i8_dynamic_rows(%input: tensor<16xi8>,
                                       %weights: tensor<?x16xi8>) -> tensor<?xi32> {
  %out = ks.matvec_i8 %input, %weights : tensor<16xi8>, tensor<?x16xi8> -> tensor<?xi32>
  return %out : tensor<?xi32>
}
