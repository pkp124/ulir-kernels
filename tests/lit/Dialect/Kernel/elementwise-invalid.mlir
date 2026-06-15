// RUN: %ks-opt %s -split-input-file -verify-diagnostics

// -----

func.func @add_unranked_lhs(%lhs: tensor<*xf32>, %rhs: tensor<4xf32>)
    -> tensor<4xf32> {
  // expected-error @+1 {{'ks.add' op operands and result must be ranked tensors}}
  %0 = ks.add %lhs, %rhs : tensor<*xf32>, tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @mul_non_float(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>)
    -> tensor<4xi32> {
  // expected-error @+1 {{'ks.mul' op operand #0 must be tensor of 16-bit float or bfloat16 type or 32-bit float or 64-bit float values}}
  %0 = ks.mul %lhs, %rhs : tensor<4xi32>, tensor<4xi32> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

func.func @add_element_type_mismatch(%lhs: tensor<4xf32>, %rhs: tensor<4xf16>)
    -> tensor<4xf32> {
  // expected-error @+1 {{'ks.add' op lhs, rhs, and result element types must match}}
  %0 = ks.add %lhs, %rhs : tensor<4xf32>, tensor<4xf16> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @mul_shape_not_broadcastable(%lhs: tensor<2x3xf32>,
                                       %rhs: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{'ks.mul' op operand shapes must be broadcast-compatible with result shape}}
  %0 = ks.mul %lhs, %rhs : tensor<2x3xf32>, tensor<2x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @add_result_shape_wrong(%lhs: tensor<2x1xf32>,
                                  %rhs: tensor<2x4xf32>)
    -> tensor<2x1xf32> {
  // expected-error @+1 {{'ks.add' op result shape must be the broadcasted operand shape}}
  %0 = ks.add %lhs, %rhs : tensor<2x1xf32>, tensor<2x4xf32> -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}
