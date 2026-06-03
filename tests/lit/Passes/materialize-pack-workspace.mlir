// RUN: %ks-opt %s --ks-pack --ks-materialize-pack-workspace | %FileCheck %s

// CHECK-LABEL: func.func @materialize_b_pack_workspace
// CHECK-SAME: memref<?xi8>
// CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
// CHECK: %[[PACKED:.*]] = memref.view %arg3[%[[OFFSET]]][] : memref<?xi8> to memref<4x256x32xf32>
// CHECK: %[[B_BUF:.*]] = bufferization.to_buffer %{{.*}} : tensor<256x128xf32> to memref<256x128xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[B_BUF]]
// CHECK-SAME: outs(%[[PACKED]]
// CHECK: %[[PACKED_TENSOR:.*]] = bufferization.to_tensor %[[PACKED]] restrict : memref<4x256x32xf32> to tensor<4x256x32xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %[[PACKED_TENSOR]]
// CHECK-NOT: linalg.pack
// CHECK-NOT: linalg.unpack
// CHECK-NOT: memref.alloc
func.func @materialize_b_pack_workspace(
    %A: memref<64x256xf32>,
    %B: memref<256x128xf32>,
    %C: memref<64x128xf32>,
    %workspace: memref<?xi8>) {
  %At = bufferization.to_tensor %A restrict : memref<64x256xf32> to tensor<64x256xf32>
  %Bt = bufferization.to_tensor %B restrict : memref<256x128xf32> to tensor<256x128xf32>
  %Ct = bufferization.to_tensor %C restrict writable : memref<64x128xf32> to tensor<64x128xf32>
  %D = linalg.matmul ins(%At, %Bt : tensor<64x256xf32>, tensor<256x128xf32>)
                     outs(%Ct : tensor<64x128xf32>) -> tensor<64x128xf32>
  %Dm = bufferization.to_buffer %D : tensor<64x128xf32> to memref<64x128xf32>
  memref.copy %Dm, %C : memref<64x128xf32> to memref<64x128xf32>
  return
}

// CHECK-LABEL: func.func @materialize_custom_alignment
// CHECK-SAME: memref<?xi8>
// CHECK: memref.view %arg3
func.func @materialize_custom_alignment(
    %A: memref<32x256xf32>,
    %B: memref<256x64xf32>,
    %C: memref<32x64xf32>,
    %workspace: memref<?xi8>) {
  %At = bufferization.to_tensor %A restrict : memref<32x256xf32> to tensor<32x256xf32>
  %Bt = bufferization.to_tensor %B restrict : memref<256x64xf32> to tensor<256x64xf32>
  %Ct = bufferization.to_tensor %C restrict writable : memref<32x64xf32> to tensor<32x64xf32>
  %D = linalg.matmul ins(%At, %Bt : tensor<32x256xf32>, tensor<256x64xf32>)
                     outs(%Ct : tensor<32x64xf32>) -> tensor<32x64xf32>
  %Dm = bufferization.to_buffer %D : tensor<32x64xf32> to memref<32x64xf32>
  memref.copy %Dm, %C : memref<32x64xf32> to memref<32x64xf32>
  return
}
