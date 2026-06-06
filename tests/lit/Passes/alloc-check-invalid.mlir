// RUN: %ks-opt %s --ks-alloc-check -split-input-file -verify-diagnostics

func.func @rejects_alloc() {
  // expected-error @+1 {{KernelSmith generated kernels must not contain memref.alloc; use caller-provided workspace buffers}}
  %buffer = memref.alloc() : memref<16xf32>
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  memref.store %zero, %buffer[%c0] : memref<16xf32>
  return
}
