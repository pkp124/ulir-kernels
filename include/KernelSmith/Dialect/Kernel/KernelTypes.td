//===----------------------------------------------------------------------===//
// KernelSmith Kernel Types
//===----------------------------------------------------------------------===//

#ifndef KERNELSMITH_DIALECT_KERNEL_KERNELTYPES_TD
#define KERNELSMITH_DIALECT_KERNEL_KERNELTYPES_TD

include "KernelSmith/Dialect/Kernel/KernelDialect.td"

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

def KS_TileType : KS_Type<"Tile", "tile"> {
  let summary = "A tile type for blocked computation";
  let description = [{
    Represents a tile/block of data used in tiled computations.
    
    Example:
    ```mlir
    !ks.tile<64x64, f32>
    ```
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );

  let assemblyFormat = "`<` $shape `,` $elementType `>`";

  let extraClassDeclaration = [{
    int64_t getNumElements() const;
    int64_t getRank() const { return getShape().size(); }
  }];
}

#endif // KERNELSMITH_DIALECT_KERNEL_KERNELTYPES_TD
