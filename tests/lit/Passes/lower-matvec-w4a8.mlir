// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func @test_matvec_w4a8
// CHECK-NOT:   ks.matvec_w4a8
// CHECK-DAG:   %[[GROUP_SIZE:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[WEIGHT_ZP:.*]] = arith.constant 2 : i32
// CHECK-DAG:   %[[INPUT_ZP:.*]] = arith.constant -1 : i32
// CHECK-DAG:   %[[INPUT_SCALE:.*]] = arith.constant 2.500000e-01 : f32
// CHECK-DAG:   %[[LOW_MASK:.*]] = arith.constant 15 : i32
// CHECK-DAG:   %[[HIGH_SHIFT:.*]] = arith.constant 4 : i32
// CHECK-DAG:   %[[SIGN_SHIFT:.*]] = arith.constant 28 : i32
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<3xf32>
// CHECK:       %[[INIT:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<3xf32>) -> tensor<3xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%{{.*}} : tensor<7xi8>)
// CHECK-SAME:    outs(%[[INIT]] : tensor<3xf32>)
// CHECK:       ^bb0(%[[INPUT:.*]]: i8, %[[ACC:.*]]: f32):
// CHECK:         %[[ROW:.*]] = linalg.index 0 : index
// CHECK:         %[[COL:.*]] = linalg.index 1 : index
// CHECK:         %[[PACKED_INDEX:.*]] = arith.divui %[[COL]], %[[TWO]] : index
// CHECK:         %[[PACKED:.*]] = tensor.extract %{{.*}}[%[[ROW]], %[[PACKED_INDEX]]] : tensor<3x4xi8>
// CHECK:         %[[PACKED_I32:.*]] = arith.extui %[[PACKED]] : i8 to i32
// CHECK:         %[[LOW:.*]] = arith.andi %[[PACKED_I32]], %[[LOW_MASK]] : i32
// CHECK:         %[[HIGH:.*]] = arith.shrui %[[PACKED_I32]], %[[HIGH_SHIFT]] : i32
// CHECK:         %[[REM:.*]] = arith.remui %[[COL]], %[[TWO]] : index
// CHECK:         %[[IS_HIGH:.*]] = arith.cmpi eq, %[[REM]], %[[ONE]] : index
// CHECK:         %[[NIBBLE:.*]] = arith.select %[[IS_HIGH]], %[[HIGH]], %[[LOW]] : i32
// CHECK:         %[[SHIFTED:.*]] = arith.shli %[[NIBBLE]], %[[SIGN_SHIFT]] : i32
// CHECK:         %[[WEIGHT_I32:.*]] = arith.shrsi %[[SHIFTED]], %[[SIGN_SHIFT]] : i32
// CHECK:         %[[CENTERED_WEIGHT:.*]] = arith.subi %[[WEIGHT_I32]], %[[WEIGHT_ZP]] : i32
// CHECK:         %[[INPUT_I32:.*]] = arith.extsi %[[INPUT]] : i8 to i32
// CHECK:         %[[CENTERED_INPUT:.*]] = arith.subi %[[INPUT_I32]], %[[INPUT_ZP]] : i32
// CHECK:         %[[INPUT_F32:.*]] = arith.sitofp %[[CENTERED_INPUT]] : i32 to f32
// CHECK:         %[[SCALED_INPUT:.*]] = arith.mulf %[[INPUT_F32]], %[[INPUT_SCALE]] : f32
// CHECK:         %[[SCALE_INDEX:.*]] = arith.divui %[[COL]], %[[GROUP_SIZE]] : index
// CHECK:         %[[WEIGHT_SCALE:.*]] = tensor.extract %{{.*}}[%[[ROW]], %[[SCALE_INDEX]]] : tensor<3x3xf32>
// CHECK:         %[[WEIGHT_F32:.*]] = arith.sitofp %[[CENTERED_WEIGHT]] : i32 to f32
// CHECK:         %[[SCALED_WEIGHT:.*]] = arith.mulf %[[WEIGHT_F32]], %[[WEIGHT_SCALE]] : f32
// CHECK:         %[[PRODUCT:.*]] = arith.mulf %[[SCALED_INPUT]], %[[SCALED_WEIGHT]] : f32
// CHECK:         %[[SUM:.*]] = arith.addf %[[ACC]], %[[PRODUCT]] : f32
// CHECK:         linalg.yield %[[SUM]] : f32
func.func @test_matvec_w4a8(%input: tensor<7xi8>,
                            %packed_weights: tensor<3x4xi8>,
                            %weight_scales: tensor<3x3xf32>)
    -> tensor<3xf32> {
  %output = ks.matvec_w4a8 %input, %packed_weights, %weight_scales
      {group_size = 3 : i64, input_scale = 2.500000e-01 : f64,
       input_zero_point = -1 : i64, weight_zero_point = 2 : i64}
      : tensor<7xi8>, tensor<3x4xi8>, tensor<3x3xf32> -> tensor<3xf32>
  return %output : tensor<3xf32>
}

// CHECK-LABEL: func @test_matvec_w4a8_dynamic_rows
// CHECK-NOT:   ks.matvec_w4a8
// CHECK:       tensor.dim %{{.*}}, %c0
// CHECK:       tensor.empty(%{{.*}}) : tensor<?xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}} : tensor<7xi8>)
// CHECK:         tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<?x4xi8>
// CHECK:         tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<?x2xf32>
func.func @test_matvec_w4a8_dynamic_rows(%input: tensor<7xi8>,
                                         %packed_weights: tensor<?x4xi8>,
                                         %weight_scales: tensor<?x2xf32>)
    -> tensor<?xf32> {
  %output = ks.matvec_w4a8 %input, %packed_weights, %weight_scales
      {group_size = 4 : i64, input_scale = 5.000000e-01 : f64}
      : tensor<7xi8>, tensor<?x4xi8>, tensor<?x2xf32> -> tensor<?xf32>
  return %output : tensor<?xf32>
}
