//===----------------------------------------------------------------------===//
// KernelSmith Kernel Operations Implementation
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace kernelsmith::ks;

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto lhsType = getLhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = getRhs().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType || !resultType)
    return emitOpError("operands and result must be ranked tensors");

  if (lhsType.getRank() != 2)
    return emitOpError("left operand must be a 2D tensor, got rank ")
           << lhsType.getRank();

  if (rhsType.getRank() != 2)
    return emitOpError("right operand must be a 2D tensor, got rank ")
           << rhsType.getRank();

  if (resultType.getRank() != 2)
    return emitOpError("result must be a 2D tensor, got rank ")
           << resultType.getRank();

  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOpError("operand element types must match");

  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  
  if (lhsK != ShapedType::kDynamic && rhsK != ShapedType::kDynamic &&
      lhsK != rhsK) {
    return emitOpError("inner dimensions must match: lhs has ")
           << lhsK << ", rhs has " << rhsK;
  }

  return success();
}

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  // TODO: Add patterns
}

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult BatchMatmulOp::verify() {
  auto lhsType = getLhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = getRhs().getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType)
    return emitOpError("operands must be ranked tensors");

  if (lhsType.getRank() < 3)
    return emitOpError("left operand must have at least 3 dimensions");

  if (rhsType.getRank() < 3)
    return emitOpError("right operand must have at least 3 dimensions");

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto filterType = getFilter().getType().dyn_cast<RankedTensorType>();

  if (!inputType || !filterType)
    return emitOpError("operands must be ranked tensors");

  if (inputType.getRank() != 4)
    return emitOpError("input must be 4D tensor (NHWC)");

  if (filterType.getRank() != 4)
    return emitOpError("filter must be 4D tensor (HWIO)");

  return success();
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

LogicalResult ScaledDotProductAttentionOp::verify() {
  auto queryType = getQuery().getType().dyn_cast<RankedTensorType>();
  auto keyType = getKey().getType().dyn_cast<RankedTensorType>();
  auto valueType = getValue().getType().dyn_cast<RankedTensorType>();

  if (!queryType || !keyType || !valueType)
    return emitOpError("query, key, and value must be ranked tensors");

  if (queryType.getRank() != 3)
    return emitOpError("query must be 3D tensor (batch, seq, dim)");

  if (keyType.getRank() != 3)
    return emitOpError("key must be 3D tensor");

  if (valueType.getRank() != 3)
    return emitOpError("value must be 3D tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//

LogicalResult LayerNormOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  
  if (!inputType)
    return emitOpError("input must be a ranked tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KernelSmith/Dialect/Kernel/KernelOps.cpp.inc"
