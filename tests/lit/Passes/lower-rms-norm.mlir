// RUN: %ks-opt %s --ks-lower-to-linalg | %FileCheck %s

// CHECK-LABEL: func @test_rms_norm_2d
// CHECK-NOT:   ks.rms_norm
// Sum-of-squares reduction over the trailing dimension, f32 accumulation.
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       linalg.fill ins(%[[ZERO]]
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK:       ^bb0(%[[IN:.*]]: f32, %[[ACC:.*]]: f32):
// CHECK:         %[[SQ:.*]] = arith.mulf %[[IN]], %[[IN]] : f32
// CHECK:         %[[SUM:.*]] = arith.addf %[[ACC]], %[[SQ]] : f32
// CHECK:         linalg.yield %[[SUM]] : f32
// scale = rsqrt(ssq / inner + eps)
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel"]
// CHECK:         %[[MEAN:.*]] = arith.divf
// CHECK:         %[[MEPS:.*]] = arith.addf %[[MEAN]], %{{.*}} : f32
// CHECK:         %[[SCALE:.*]] = math.rsqrt %[[MEPS]] : f32
// CHECK:         linalg.yield %[[SCALE]] : f32
// output = input * scale * weight
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:       ^bb0(%[[X:.*]]: f32, %[[S:.*]]: f32, %[[W:.*]]: f32, %{{.*}}: f32):
// CHECK:         %[[XS:.*]] = arith.mulf %[[X]], %[[S]] : f32
// CHECK:         %[[OUT:.*]] = arith.mulf %[[XS]], %[[W]] : f32
// CHECK:         linalg.yield %[[OUT]] : f32
func.func @test_rms_norm_2d(%input: tensor<2x4xf32>,
                            %weight: tensor<4xf32>) -> tensor<2x4xf32> {
  %out = ks.rms_norm %input, %weight {eps = 1.000000e-05 : f32}
      : tensor<2x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

// -----

// A 3D input normalizes over the trailing dimension only: two parallel outer
// dims plus a reduction over the last.
// CHECK-LABEL: func @test_rms_norm_3d
// CHECK-NOT:   ks.rms_norm
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:       math.rsqrt
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel"]
func.func @test_rms_norm_3d(%input: tensor<2x3x8xf32>,
                            %weight: tensor<8xf32>) -> tensor<2x3x8xf32> {
  %out = ks.rms_norm %input, %weight {eps = 1.000000e-05 : f32}
      : tensor<2x3x8xf32>, tensor<8xf32> -> tensor<2x3x8xf32>
  return %out : tensor<2x3x8xf32>
}

// -----

// Dynamic trailing dimension: the element count is read at runtime and
// converted to f32 for the mean division.
// CHECK-LABEL: func @test_rms_norm_dynamic
// CHECK-NOT:   ks.rms_norm
// CHECK:       tensor.dim
// CHECK:       arith.index_cast
// CHECK:       arith.sitofp
// CHECK:       math.rsqrt
func.func @test_rms_norm_dynamic(%input: tensor<2x?xf32>,
                                 %weight: tensor<?xf32>) -> tensor<2x?xf32> {
  %out = ks.rms_norm %input, %weight {eps = 1.000000e-05 : f32}
      : tensor<2x?xf32>, tensor<?xf32> -> tensor<2x?xf32>
  return %out : tensor<2x?xf32>
}

// -----

// Narrow floating-point inputs are promoted to f32 for reduction and scaling,
// then truncated back to the declared result element type.
// CHECK-LABEL: func @test_rms_norm_f16
// CHECK-NOT:   ks.rms_norm
// CHECK:       linalg.generic
// CHECK:         %[[X32:.*]] = arith.extf %{{.*}} : f16 to f32
// CHECK:         %[[SQ32:.*]] = arith.mulf %[[X32]], %[[X32]] : f32
// CHECK:         arith.addf %{{.*}}, %[[SQ32]] : f32
// CHECK:       math.rsqrt
// CHECK:       linalg.generic
// CHECK:         arith.extf %{{.*}} : f16 to f32
// CHECK:         arith.truncf %{{.*}} : f32 to f16
func.func @test_rms_norm_f16(%input: tensor<2x4xf16>,
                             %weight: tensor<4xf16>) -> tensor<2x4xf16> {
  %out = ks.rms_norm %input, %weight {eps = 1.000000e-05 : f32}
      : tensor<2x4xf16>, tensor<4xf16> -> tensor<2x4xf16>
  return %out : tensor<2x4xf16>
}
