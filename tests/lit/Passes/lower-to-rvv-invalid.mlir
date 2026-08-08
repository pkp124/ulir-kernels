// RUN: %ks-opt %s --ks-lower-to-rvv -verify-diagnostics

func.func @large_multi_reduction(%input: vector<513x2xf32>,
                                 %acc: vector<513xf32>)
    -> vector<513xf32> {
  // expected-error @+1 {{--ks-lower-to-rvv does not support vector.multi_reduction with 513 result elements; maximum is 512; tile the operation to bound each vector reduction result}}
  %result = vector.multi_reduction <add>, %input, %acc [1]
      : vector<513x2xf32> to vector<513xf32>
  return %result : vector<513xf32>
}
