// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// Default axis (-1) normalizes over the trailing dimension: max-reduce,
// exp(x - max), sum-reduce, divide.
// CHECK-LABEL: func @test_softmax_2d
// CHECK-NOT:   ks.softmax
// CHECK:       %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:       linalg.fill ins(%[[NINF]]
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK:         arith.maximumf
// exp(input - max)
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:         %[[SHIFT:.*]] = arith.subf
// CHECK:         math.exp %[[SHIFT]]
// sum-reduce of the exponentials
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       linalg.fill ins(%[[ZERO]]
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK:         arith.addf
// divide
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:         arith.divf
func.func @test_softmax_2d(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %out = ks.softmax %input : tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----

// An explicit non-trailing axis reduces over the middle dimension.
// CHECK-LABEL: func @test_softmax_axis1
// CHECK-NOT:   ks.softmax
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:         arith.maximumf
// CHECK:       math.exp
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:         arith.addf
// CHECK:       arith.divf
func.func @test_softmax_axis1(%input: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %out = ks.softmax %input {axis = 1 : i64} : tensor<2x3x4xf32>
  return %out : tensor<2x3x4xf32>
}

// -----

// Dynamic trailing dimension still lowers; reduced tensors source their
// dynamic sizes from the input.
// CHECK-LABEL: func @test_softmax_dynamic
// CHECK-NOT:   ks.softmax
// CHECK:       tensor.dim
// CHECK:       arith.maximumf
// CHECK:       math.exp
// CHECK:       arith.addf
// CHECK:       arith.divf
func.func @test_softmax_dynamic(%input: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %out = ks.softmax %input : tensor<2x?xf32>
  return %out : tensor<2x?xf32>
}
