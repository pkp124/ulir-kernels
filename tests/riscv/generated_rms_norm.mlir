// Self-checking generated-kernel test for the complete KS-to-RVV pipeline.
//
// Compile and run:
//   MLIR_TRANSLATE=mlir-translate-21 LLC=llc-21 \
//     scripts/compile-rvv.sh tests/riscv/generated_rms_norm.mlir \
//     /tmp/generated_rms_norm.o
//   riscv64-linux-gnu-gcc -static /tmp/generated_rms_norm.o \
//     -o /tmp/generated_rms_norm
//   python tests/riscv_runner.py --binary /tmp/generated_rms_norm \
//     --vlens 256 512
//
// Exit code 0 = PASS, 1 = FAIL.

func.func @main() -> i32 {
  %input = arith.constant dense<[
    [3.0, 4.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0]
  ]> : tensor<2x4xf32>
  %weight = arith.constant dense<[1.0, 0.5, 2.0, 1.0]>
      : tensor<4xf32>

  %output = ks.rms_norm %input, %weight {eps = 1.000000e-05 : f32}
      : tensor<2x4xf32>, tensor<4xf32> -> tensor<2x4xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %actual00 = tensor.extract %output[%c0, %c0] : tensor<2x4xf32>
  %actual01 = tensor.extract %output[%c0, %c1] : tensor<2x4xf32>
  %actual12 = tensor.extract %output[%c1, %c2] : tensor<2x4xf32>
  %actual13 = tensor.extract %output[%c1, %c3] : tensor<2x4xf32>

  %expected00 = arith.constant 1.199999e+00 : f32
  %expected01 = arith.constant 7.999994e-01 : f32
  %expected12 = arith.constant 1.999990e+00 : f32
  %expected13 = arith.constant -9.999950e-01 : f32
  %tolerance = arith.constant 1.000000e-04 : f32

  %lo00 = arith.subf %expected00, %tolerance : f32
  %hi00 = arith.addf %expected00, %tolerance : f32
  %above00 = arith.cmpf ogt, %actual00, %lo00 : f32
  %below00 = arith.cmpf olt, %actual00, %hi00 : f32
  %ok00 = arith.andi %above00, %below00 : i1

  %lo01 = arith.subf %expected01, %tolerance : f32
  %hi01 = arith.addf %expected01, %tolerance : f32
  %above01 = arith.cmpf ogt, %actual01, %lo01 : f32
  %below01 = arith.cmpf olt, %actual01, %hi01 : f32
  %ok01 = arith.andi %above01, %below01 : i1

  %lo12 = arith.subf %expected12, %tolerance : f32
  %hi12 = arith.addf %expected12, %tolerance : f32
  %above12 = arith.cmpf ogt, %actual12, %lo12 : f32
  %below12 = arith.cmpf olt, %actual12, %hi12 : f32
  %ok12 = arith.andi %above12, %below12 : i1

  %lo13 = arith.subf %expected13, %tolerance : f32
  %hi13 = arith.addf %expected13, %tolerance : f32
  %above13 = arith.cmpf ogt, %actual13, %lo13 : f32
  %below13 = arith.cmpf olt, %actual13, %hi13 : f32
  %ok13 = arith.andi %above13, %below13 : i1

  %ok0 = arith.andi %ok00, %ok01 : i1
  %ok1 = arith.andi %ok12, %ok13 : i1
  %ok = arith.andi %ok0, %ok1 : i1
  %pass = arith.constant 0 : i32
  %fail = arith.constant 1 : i32
  %result = arith.select %ok, %pass, %fail : i32
  return %result : i32
}
