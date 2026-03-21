// Standalone MLIR RVV test: vector relu using arith.maximumf.
//
// Compile and run:
//   ks-opt tests/riscv/relu_rvv.mlir --ks-lower-to-rvv -o /tmp/relu_llvm.mlir
//   mlir-translate --mlir-to-llvmir /tmp/relu_llvm.mlir -o /tmp/relu.ll
//   llc -march=riscv64 -mattr=+v,+zve64d -float-abi=double \
//       -filetype=obj /tmp/relu.ll -o /tmp/relu.o
//   riscv64-linux-gnu-gcc -static /tmp/relu.o -o /tmp/relu_test
//   qemu-riscv64 -cpu rv64,v=true,vlen=256 /tmp/relu_test
//
// Exit code 0 = PASS, 1 = FAIL.

func.func @main() -> i32 {
  // Allocate 8-element input and output buffers on the stack.
  %buf_in  = memref.alloca() : memref<8xf32>
  %buf_out = memref.alloca() : memref<8xf32>

  // Index constants.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index

  // Input: [-2, -1, -0.5, 0, 0.5, 1, 2, 4]
  %v0 = arith.constant -2.0 : f32
  %v1 = arith.constant -1.0 : f32
  %v2 = arith.constant -0.5 : f32
  %v3 = arith.constant  0.0 : f32
  %v4 = arith.constant  0.5 : f32
  %v5 = arith.constant  1.0 : f32
  %v6 = arith.constant  2.0 : f32
  %v7 = arith.constant  4.0 : f32

  memref.store %v0, %buf_in[%c0] : memref<8xf32>
  memref.store %v1, %buf_in[%c1] : memref<8xf32>
  memref.store %v2, %buf_in[%c2] : memref<8xf32>
  memref.store %v3, %buf_in[%c3] : memref<8xf32>
  memref.store %v4, %buf_in[%c4] : memref<8xf32>
  memref.store %v5, %buf_in[%c5] : memref<8xf32>
  memref.store %v6, %buf_in[%c6] : memref<8xf32>
  memref.store %v7, %buf_in[%c7] : memref<8xf32>

  // --- Vector relu: out = max(in, 0) ---
  %pad   = arith.constant 0.0 : f32
  %zeros = arith.constant dense<0.0> : vector<8xf32>
  %in_v  = vector.transfer_read %buf_in[%c0], %pad
               : memref<8xf32>, vector<8xf32>
  %out_v = arith.maximumf %in_v, %zeros : vector<8xf32>
  vector.transfer_write %out_v, %buf_out[%c0]
               : vector<8xf32>, memref<8xf32>

  // --- Validate: first 4 outputs must be 0, last 4 must equal inputs ---
  %e0 = arith.constant 0.0 : f32
  %e4 = arith.constant 0.5 : f32
  %e5 = arith.constant 1.0 : f32
  %e6 = arith.constant 2.0 : f32
  %e7 = arith.constant 4.0 : f32
  %eps = arith.constant 1.0e-6 : f32

  %o0 = memref.load %buf_out[%c0] : memref<8xf32>
  %o1 = memref.load %buf_out[%c1] : memref<8xf32>
  %o2 = memref.load %buf_out[%c2] : memref<8xf32>
  %o3 = memref.load %buf_out[%c3] : memref<8xf32>
  %o4 = memref.load %buf_out[%c4] : memref<8xf32>
  %o5 = memref.load %buf_out[%c5] : memref<8xf32>
  %o6 = memref.load %buf_out[%c6] : memref<8xf32>
  %o7 = memref.load %buf_out[%c7] : memref<8xf32>

  // Check each output against expected using |got - exp| < eps.
  %d0 = arith.subf %o0, %e0 : f32
  %d1 = arith.subf %o1, %e0 : f32
  %d2 = arith.subf %o2, %e0 : f32
  %d3 = arith.subf %o3, %e0 : f32
  %d4 = arith.subf %o4, %e4 : f32
  %d5 = arith.subf %o5, %e5 : f32
  %d6 = arith.subf %o6, %e6 : f32
  %d7 = arith.subf %o7, %e7 : f32

  %a0 = math.absf %d0 : f32
  %a1 = math.absf %d1 : f32
  %a2 = math.absf %d2 : f32
  %a3 = math.absf %d3 : f32
  %a4 = math.absf %d4 : f32
  %a5 = math.absf %d5 : f32
  %a6 = math.absf %d6 : f32
  %a7 = math.absf %d7 : f32

  %ok0 = arith.cmpf olt, %a0, %eps : f32
  %ok1 = arith.cmpf olt, %a1, %eps : f32
  %ok2 = arith.cmpf olt, %a2, %eps : f32
  %ok3 = arith.cmpf olt, %a3, %eps : f32
  %ok4 = arith.cmpf olt, %a4, %eps : f32
  %ok5 = arith.cmpf olt, %a5, %eps : f32
  %ok6 = arith.cmpf olt, %a6, %eps : f32
  %ok7 = arith.cmpf olt, %a7, %eps : f32

  %all01 = arith.andi %ok0, %ok1 : i1
  %all23 = arith.andi %ok2, %ok3 : i1
  %all45 = arith.andi %ok4, %ok5 : i1
  %all67 = arith.andi %ok6, %ok7 : i1
  %all03 = arith.andi %all01, %all23 : i1
  %all47 = arith.andi %all45, %all67 : i1
  %all   = arith.andi %all03, %all47 : i1

  %pass = arith.constant 0 : i32
  %fail = arith.constant 1 : i32
  %ret  = arith.select %all, %pass, %fail : i32
  return %ret : i32
}
