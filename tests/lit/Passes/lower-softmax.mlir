// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s
// RUN: %ks-opt %s --ks-lower-to-linalg --ks-vectorize | %FileCheck %s --check-prefix=VECTOR
// RUN: %ks-opt %s --ks-lower-to-linalg --ks-vectorize --ks-lower-to-rvv -o %t
// RUN: %mlir-translate --mlir-to-llvmir %t -o /dev/null

// The default trailing axis lowers to the numerically stable sequence:
// max reduction, subtract-and-exp, sum reduction, and divide.
// CHECK-LABEL: func @softmax_trailing_axis
// VECTOR-LABEL: func @softmax_trailing_axis
// VECTOR:       vector.multi_reduction <maximumf>
// VECTOR:       math.exp %{{.*}} : vector<2x4xf32>
// VECTOR:       vector.multi_reduction <add>
// VECTOR:       arith.divf %{{.*}}, %{{.*}} : vector<2x4xf32>
// CHECK-NOT:   ks.softmax
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:       linalg.fill ins(%[[NINF]]
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK:         arith.maximumf
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:         %[[SHIFTED:.*]] = arith.subf
// CHECK:         math.exp %[[SHIFTED]]
// CHECK:       linalg.fill ins(%[[ZERO]]
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK:         arith.addf
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:         arith.divf
func.func @softmax_trailing_axis(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %output = ks.softmax %input : tensor<2x4xf32>
  return %output : tensor<2x4xf32>
}

// An explicit middle axis makes only that loop a reduction.
// CHECK-LABEL: func @softmax_middle_axis
// CHECK-NOT:   ks.softmax
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:         arith.maximumf
// CHECK:       math.exp
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:         arith.addf
// CHECK:       arith.divf
func.func @softmax_middle_axis(
    %input: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %output = ks.softmax %input {axis = 1 : i64} : tensor<2x3x4xf32>
  return %output : tensor<2x3x4xf32>
}

// A negative axis is normalized relative to the input rank.
// CHECK-LABEL: func @softmax_negative_axis
// CHECK-NOT:   ks.softmax
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["reduction", "parallel"]
// CHECK:         arith.maximumf
// CHECK:       math.exp
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["reduction", "parallel"]
// CHECK:         arith.addf
func.func @softmax_negative_axis(
    %input: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %output = ks.softmax %input {axis = -2 : i64} : tensor<3x4xf32>
  return %output : tensor<3x4xf32>
}

// Dynamic dimensions are sourced from the input for intermediate tensors.
// CHECK-LABEL: func @softmax_dynamic
// CHECK-NOT:   ks.softmax
// CHECK:       tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xf32>
// CHECK:       arith.maximumf
// CHECK:       math.exp
// CHECK:       arith.addf
// CHECK:       arith.divf
func.func @softmax_dynamic(
    %input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %output = ks.softmax %input : tensor<?x?xf32>
  return %output : tensor<?x?xf32>
}

// Narrow inputs use f32 intermediates for both reductions and exp/divide.
// CHECK-LABEL: func @softmax_f16
// CHECK-NOT:   ks.softmax
// CHECK:       arith.constant 0xFF800000 : f32
// CHECK:       arith.extf %{{.*}} : f16 to f32
// CHECK:       math.exp %{{.*}} : f32
// CHECK:       arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:       arith.divf %{{.*}}, %{{.*}} : f32
// CHECK:       arith.truncf %{{.*}} : f32 to f16
func.func @softmax_f16(%input: tensor<2x4xf16>) -> tensor<2x4xf16> {
  %output = ks.softmax %input : tensor<2x4xf16>
  return %output : tensor<2x4xf16>
}
