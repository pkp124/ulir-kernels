//===----------------------------------------------------------------------===//
// KernelSmith Kernel Operations Implementation
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace kernelsmith::ks;

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

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
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!lhsType || !rhsType || !resultType)
    return emitOpError("operands and result must be ranked tensors");

  if (lhsType.getRank() < 3)
    return emitOpError("left operand must have at least 3 dimensions");

  if (rhsType.getRank() < 3)
    return emitOpError("right operand must have at least 3 dimensions");

  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOpError("operand element types must match");

  // Check inner dimensions: lhs[..., K] == rhs[..., K, N]
  int64_t lhsK = lhsType.getDimSize(lhsType.getRank() - 1);
  int64_t rhsK = rhsType.getDimSize(rhsType.getRank() - 2);

  if (lhsK != ShapedType::kDynamic && rhsK != ShapedType::kDynamic &&
      lhsK != rhsK) {
    return emitOpError("inner dimensions must match: lhs has ")
           << lhsK << ", rhs has " << rhsK;
  }

  // Check batch dimensions match
  if (lhsType.getRank() != rhsType.getRank())
    return emitOpError("operands must have the same rank");

  int64_t rank = lhsType.getRank();
  for (int64_t i = 0; i < rank - 2; ++i) {
    int64_t lhsDim = lhsType.getDimSize(i);
    int64_t rhsDim = rhsType.getDimSize(i);
    if (lhsDim != ShapedType::kDynamic && rhsDim != ShapedType::kDynamic &&
        lhsDim != rhsDim) {
      return emitOpError("batch dimension ")
             << i << " must match: lhs has " << lhsDim << ", rhs has "
             << rhsDim;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = dyn_cast<RankedTensorType>(getFilter().getType());

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
  auto queryType = dyn_cast<RankedTensorType>(getQuery().getType());
  auto keyType = dyn_cast<RankedTensorType>(getKey().getType());
  auto valueType = dyn_cast<RankedTensorType>(getValue().getType());

  if (!queryType || !keyType || !valueType)
    return emitOpError("query, key, and value must be ranked tensors");

  if (queryType.getRank() != 3)
    return emitOpError("query must be 3D tensor (batch, seq, dim)");

  if (keyType.getRank() != 3)
    return emitOpError("key must be 3D tensor");

  if (valueType.getRank() != 3)
    return emitOpError("value must be 3D tensor");

  // Element types must match across Q, K, V
  if (queryType.getElementType() != keyType.getElementType() ||
      queryType.getElementType() != valueType.getElementType())
    return emitOpError("query, key, and value element types must match");

  // Query head dim (dim 2) must match Key head dim (dim 2) for Q * K^T
  int64_t queryDim = queryType.getDimSize(2);
  int64_t keyDim = keyType.getDimSize(2);
  if (queryDim != ShapedType::kDynamic && keyDim != ShapedType::kDynamic &&
      queryDim != keyDim) {
    return emitOpError("query head dimension (")
           << queryDim << ") must match key head dimension (" << keyDim << ")";
  }

  // Key seq length (dim 1) must match Value seq length (dim 1)
  int64_t keySeq = keyType.getDimSize(1);
  int64_t valueSeq = valueType.getDimSize(1);
  if (keySeq != ShapedType::kDynamic && valueSeq != ShapedType::kDynamic &&
      keySeq != valueSeq) {
    return emitOpError("key sequence length (")
           << keySeq << ") must match value sequence length (" << valueSeq
           << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//

LogicalResult LayerNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());

  if (!inputType)
    return emitOpError("input must be a ranked tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());

  if (!inputType)
    return emitOpError("input must be a ranked tensor");

  int64_t rank = inputType.getRank();
  int64_t axis = getAxis();

  if (axis < -rank || axis >= rank) {
    return emitOpError("axis ")
           << axis << " is out of range for tensor of rank " << rank;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//

LogicalResult RMSNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());

  if (!inputType || !weightType)
    return emitOpError("input and weight must be ranked tensors");

  if (inputType.getRank() < 1)
    return emitOpError("input must have at least 1 dimension");

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceSumOp
//===----------------------------------------------------------------------===//

/// Shared verification logic for reduction operations.
static LogicalResult
verifyReductionAxes(Operation *op, RankedTensorType inputType, ArrayAttr axes) {
  int64_t rank = inputType.getRank();
  llvm::SmallSet<int64_t, 4> seen;

  for (auto axisAttr : axes) {
    int64_t axis = cast<IntegerAttr>(axisAttr).getInt();

    if (axis < -rank || axis >= rank) {
      return op->emitOpError("axis ")
             << axis << " is out of range for tensor of rank " << rank;
    }

    // Normalize negative axis for duplicate check
    int64_t normalized = axis < 0 ? axis + rank : axis;
    if (!seen.insert(normalized).second) {
      return op->emitOpError("duplicate axis ") << axis;
    }
  }

  return success();
}

LogicalResult ReduceSumOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());

  if (!inputType)
    return emitOpError("input must be a ranked tensor");

  return verifyReductionAxes(getOperation(), inputType, getAxes());
}

//===----------------------------------------------------------------------===//
// ReduceMaxOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceMaxOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());

  if (!inputType)
    return emitOpError("input must be a ranked tensor");

  return verifyReductionAxes(getOperation(), inputType, getAxes());
}

//===----------------------------------------------------------------------===//
// TableGen Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KernelSmith/Dialect/Kernel/KernelOps.cpp.inc"
