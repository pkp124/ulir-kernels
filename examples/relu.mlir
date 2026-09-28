// ReLU activation.
//
//   build/bin/ks-opt examples/relu.mlir --ks-lower-activations

func.func @relu_example(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %output = ks.relu %input : tensor<1024xf32>
  return %output : tensor<1024xf32>
}
