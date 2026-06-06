// RUN: %ks-opt %s --ks-alloc-check | %FileCheck %s

// CHECK-LABEL: func.func @uses_caller_buffers
// CHECK-NOT:   memref.alloc
func.func @uses_caller_buffers(%input: memref<8xf32>,
                               %output: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %value = memref.load %input[%c0] : memref<8xf32>
  memref.store %value, %output[%c0] : memref<8xf32>
  return
}
