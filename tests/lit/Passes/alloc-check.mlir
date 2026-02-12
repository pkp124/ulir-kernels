// RUN: %ks-opt %s -split-input-file -verify-diagnostics --ks-alloc-check

// ============================================================
// No alloc — should pass cleanly
// ============================================================

func.func @test_no_alloc(%arg0: memref<64x64xf32>) -> memref<64x64xf32> {
  return %arg0 : memref<64x64xf32>
}

// -----

// ============================================================
// Has memref.alloc — should fail
// ============================================================

func.func @test_has_alloc() -> memref<64x64xf32> {
  // expected-error @+1 {{unexpected memref.alloc: all buffers must be function arguments (zero internal malloc)}}
  %0 = memref.alloc() : memref<64x64xf32>
  return %0 : memref<64x64xf32>
}

// -----

// ============================================================
// Multiple allocs — each should emit an error
// ============================================================

func.func @test_multiple_allocs() -> (memref<32xf32>, memref<64xf32>) {
  // expected-error @+1 {{unexpected memref.alloc}}
  %0 = memref.alloc() : memref<32xf32>
  // expected-error @+1 {{unexpected memref.alloc}}
  %1 = memref.alloc() : memref<64xf32>
  return %0, %1 : memref<32xf32>, memref<64xf32>
}
