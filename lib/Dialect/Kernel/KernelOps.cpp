//===----------------------------------------------------------------------===//
// KernelSmith Kernel Operations Implementation
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <optional>

using namespace mlir;
using namespace kernelsmith::ks;

static bool compatibleShapes(RankedTensorType inputType,
                             RankedTensorType resultType) {
  if (inputType.getRank() != resultType.getRank())
    return false;

  for (auto [inputDim, resultDim] :
       llvm::zip(inputType.getShape(), resultType.getShape())) {
    if (inputDim != ShapedType::kDynamic &&
        resultDim != ShapedType::kDynamic && inputDim != resultDim) {
      return false;
    }
  }

  return true;
}

static bool compatibleWithStaticShape(RankedTensorType type,
                                      ArrayRef<int64_t> shape) {
  if (type.getRank() != static_cast<int64_t>(shape.size()))
    return false;

  for (auto [actualDim, expectedDim] : llvm::zip(type.getShape(), shape)) {
    if (actualDim != ShapedType::kDynamic && actualDim != expectedDim)
      return false;
  }

  return true;
}

static LogicalResult verifyFloatingPointElementType(Operation *op,
                                                    RankedTensorType type,
                                                    StringRef tensorName) {
  if (!isa<FloatType>(type.getElementType()))
    return op->emitOpError(tensorName)
           << " element type must be floating-point";

  return success();
}

static LogicalResult verifyMatchingElementTypes(Operation *op,
                                                ArrayRef<RankedTensorType> types,
                                                StringRef tensorNames) {
  Type elementType = types.front().getElementType();
  for (RankedTensorType type : types.drop_front()) {
    if (type.getElementType() != elementType)
      return op->emitOpError(tensorNames) << " element types must match";
  }

  return success();
}

static LogicalResult verifyElementwiseBroadcast(Operation *op,
                                                RankedTensorType lhsType,
                                                RankedTensorType rhsType,
                                                RankedTensorType resultType) {
  int64_t resultRank = resultType.getRank();
  int64_t expectedRank = std::max(lhsType.getRank(), rhsType.getRank());
  if (resultRank != expectedRank)
    return op->emitOpError(
        "result shape must be the broadcasted operand shape");

  auto getAlignedDim = [resultRank](RankedTensorType type,
                                    int64_t resultDim) -> int64_t {
    int64_t offset = resultRank - type.getRank();
    if (resultDim < offset)
      return 1;
    return type.getDimSize(resultDim - offset);
  };

  for (int64_t dim = 0; dim < resultRank; ++dim) {
    int64_t lhsDim = getAlignedDim(lhsType, dim);
    int64_t rhsDim = getAlignedDim(rhsType, dim);
    bool lhsKnownNonOne =
        lhsDim != ShapedType::kDynamic && lhsDim != 1;
    bool rhsKnownNonOne =
        rhsDim != ShapedType::kDynamic && rhsDim != 1;
    if (lhsKnownNonOne && rhsKnownNonOne && lhsDim != rhsDim) {
      return op->emitOpError(
          "operand shapes must be broadcast-compatible with result shape");
    }
  }

  for (int64_t dim = 0; dim < resultRank; ++dim) {
    int64_t lhsDim = getAlignedDim(lhsType, dim);
    int64_t rhsDim = getAlignedDim(rhsType, dim);
    int64_t resultDim = resultType.getDimSize(dim);
    if (resultDim == ShapedType::kDynamic)
      continue;

    std::optional<int64_t> expectedDim;
    for (int64_t operandDim : {lhsDim, rhsDim}) {
      if (operandDim == ShapedType::kDynamic)
        continue;
      if (operandDim != 1) {
        expectedDim = operandDim;
        break;
      }
      if (!expectedDim)
        expectedDim = 1;
    }

    if (expectedDim && resultDim != *expectedDim)
      return op->emitOpError(
          "result shape must be the broadcasted operand shape");
  }

  return success();
}

static LogicalResult verifyPositiveFiniteFloatAttr(Operation *op,
                                                   FloatAttr attr,
                                                   StringRef attrName) {
  double value = attr.getValueAsDouble();
  if (!std::isfinite(value) || value <= 0.0)
    return op->emitOpError(attrName) << " must be positive and finite";

  return success();
}

static LogicalResult verifyPositiveFiniteScale(Operation *op,
                                               FloatAttr scaleAttr) {
  return verifyPositiveFiniteFloatAttr(op, scaleAttr, "scale");
}

static LogicalResult verifyZeroPointFits(Operation *op,
                                         IntegerType integerType,
                                         int64_t zeroPoint) {
  unsigned width = integerType.getWidth();

  if (integerType.isUnsigned()) {
    if (zeroPoint < 0)
      return op->emitOpError("zero_point ")
             << zeroPoint << " does not fit in integer element type "
             << integerType;

    if (width < 64 && static_cast<uint64_t>(zeroPoint) >= (1ULL << width)) {
      return op->emitOpError("zero_point ")
             << zeroPoint << " does not fit in integer element type "
             << integerType;
    }
    return success();
  }

  if (width >= 64)
    return success();

  int64_t min = -(1LL << (width - 1));
  int64_t max = (1LL << (width - 1)) - 1;
  if (zeroPoint < min || zeroPoint > max) {
    return op->emitOpError("zero_point ")
           << zeroPoint << " does not fit in integer element type "
           << integerType;
  }

  return success();
}

static LogicalResult verifySignedI8ZeroPointFits(Operation *op,
                                                 StringRef attrName,
                                                 int64_t zeroPoint) {
  constexpr int64_t minI8 = -128;
  constexpr int64_t maxI8 = 127;
  if (zeroPoint < minI8 || zeroPoint > maxI8)
    return op->emitOpError(attrName)
           << " " << zeroPoint << " does not fit in i8";

  return success();
}

static LogicalResult verifySignedI4ZeroPointFits(Operation *op,
                                                 StringRef attrName,
                                                 int64_t zeroPoint) {
  constexpr int64_t minI4 = -8;
  constexpr int64_t maxI4 = 7;
  if (zeroPoint < minI4 || zeroPoint > maxI4)
    return op->emitOpError(attrName)
           << " " << zeroPoint << " does not fit in i4";

  return success();
}

static LogicalResult verifySignlessI8ElementType(Operation *op,
                                                 RankedTensorType type,
                                                 StringRef tensorName) {
  auto elementType = dyn_cast<IntegerType>(type.getElementType());
  if (!elementType || !elementType.isSignless() || elementType.getWidth() != 8)
    return op->emitOpError(tensorName)
           << " element type must be signless i8";

  return success();
}

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

  if (resultType.getElementType() != lhsType.getElementType())
    return emitOpError("result element type must match operand element type");

  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);

  if (lhsK != ShapedType::kDynamic && rhsK != ShapedType::kDynamic &&
      lhsK != rhsK) {
    return emitOpError("inner dimensions must match: lhs has ")
           << lhsK << ", rhs has " << rhsK;
  }

  int64_t lhsM = lhsType.getDimSize(0);
  int64_t rhsN = rhsType.getDimSize(1);
  int64_t resultM = resultType.getDimSize(0);
  int64_t resultN = resultType.getDimSize(1);

  if (resultM != ShapedType::kDynamic && lhsM != ShapedType::kDynamic &&
      resultM != lhsM) {
    return emitOpError(
               "result row dimension must match lhs row dimension: result has ")
           << resultM << ", lhs has " << lhsM;
  }

  if (resultN != ShapedType::kDynamic && rhsN != ShapedType::kDynamic &&
      resultN != rhsN) {
    return emitOpError("result column dimension must match rhs column "
                       "dimension: result has ")
           << resultN << ", rhs has " << rhsN;
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
// AddOp / MulOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult verifyElementwiseBinaryOp(OpTy op) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());

  if (!lhsType || !rhsType || !resultType)
    return op.emitOpError("operands and result must be ranked tensors");

  if (failed(verifyFloatingPointElementType(op.getOperation(), lhsType, "lhs")))
    return failure();

  if (failed(verifyMatchingElementTypes(
          op.getOperation(), {lhsType, rhsType, resultType},
          "lhs, rhs, and result")))
    return failure();

  return verifyElementwiseBroadcast(op.getOperation(), lhsType, rhsType,
                                    resultType);
}

LogicalResult AddOp::verify() { return verifyElementwiseBinaryOp(*this); }

LogicalResult MulOp::verify() { return verifyElementwiseBinaryOp(*this); }

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//

LogicalResult LayerNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());

  if (!inputType)
    return emitOpError("input must be a ranked tensor");

  if (!weightType)
    return emitOpError("weight must be a ranked tensor");

  if (!biasType)
    return emitOpError("bias must be a ranked tensor");

  if (!outputType)
    return emitOpError("output must be a ranked tensor");

  if (failed(
          verifyFloatingPointElementType(getOperation(), inputType, "input")))
    return failure();

  if (failed(verifyMatchingElementTypes(
          getOperation(), {inputType, weightType, biasType, outputType},
          "input, weight, bias, and output")))
    return failure();

  if (!compatibleShapes(inputType, outputType))
    return emitOpError("output shape must match input shape");

  if (failed(verifyPositiveFiniteFloatAttr(getOperation(), getEpsAttr(),
                                           "eps")))
    return failure();

  SmallVector<int64_t> normalizedShape;
  normalizedShape.reserve(getNormalizedShape().size());
  for (Attribute dimAttr : getNormalizedShape()) {
    int64_t dim = cast<IntegerAttr>(dimAttr).getInt();
    if (dim <= 0)
      return emitOpError("normalized_shape dimensions must be positive");
    normalizedShape.push_back(dim);
  }

  if (normalizedShape.empty())
    return emitOpError("normalized_shape must not be empty");

  int64_t normalizedRank = normalizedShape.size();
  if (normalizedRank > inputType.getRank()) {
    return emitOpError("normalized_shape rank ")
           << normalizedRank << " exceeds input rank " << inputType.getRank();
  }

  if (!compatibleWithStaticShape(weightType, normalizedShape))
    return emitOpError("weight shape must match normalized_shape");

  if (!compatibleWithStaticShape(biasType, normalizedShape))
    return emitOpError("bias shape must match normalized_shape");

  ArrayRef<int64_t> inputTrailingShape =
      inputType.getShape().take_back(normalizedRank);
  for (auto [inputDim, normalizedDim] :
       llvm::zip(inputTrailingShape, normalizedShape)) {
    if (inputDim != ShapedType::kDynamic && inputDim != normalizedDim)
      return emitOpError(
          "input trailing dimensions must match normalized_shape");
  }

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
// QuantizeOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizeOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());

  if (!inputType || !outputType)
    return emitOpError("input and result must be ranked tensors");

  if (!isa<FloatType>(inputType.getElementType()))
    return emitOpError("input element type must be floating-point");

  auto integerType = dyn_cast<IntegerType>(outputType.getElementType());
  if (!integerType)
    return emitOpError("result element type must be integer");

  if (!compatibleShapes(inputType, outputType))
    return emitOpError("input and result shapes must match");

  if (failed(verifyPositiveFiniteScale(getOperation(), getScaleAttr())))
    return failure();

  return verifyZeroPointFits(getOperation(), integerType, getZeroPoint());
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult DequantizeOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());

  if (!inputType || !outputType)
    return emitOpError("input and result must be ranked tensors");

  auto integerType = dyn_cast<IntegerType>(inputType.getElementType());
  if (!integerType)
    return emitOpError("input element type must be integer");

  if (!isa<FloatType>(outputType.getElementType()))
    return emitOpError("result element type must be floating-point");

  if (!compatibleShapes(inputType, outputType))
    return emitOpError("input and result shapes must match");

  if (failed(verifyPositiveFiniteScale(getOperation(), getScaleAttr())))
    return failure();

  return verifyZeroPointFits(getOperation(), integerType, getZeroPoint());
}

//===----------------------------------------------------------------------===//
// DotI8Op
//===----------------------------------------------------------------------===//

LogicalResult DotI8Op::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputType || !weightType || !resultType)
    return emitOpError("operands and result must be ranked tensors");

  if (inputType.getRank() != 1)
    return emitOpError("input must be a 1D tensor");

  if (weightType.getRank() != 1)
    return emitOpError("weight must be a 1D tensor");

  if (failed(
          verifySignlessI8ElementType(getOperation(), inputType, "input")))
    return failure();

  if (failed(
          verifySignlessI8ElementType(getOperation(), weightType, "weight")))
    return failure();

  if (resultType.getRank() != 0)
    return emitOpError("result must be a rank-0 tensor");

  auto resultElementType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultElementType || resultElementType.getWidth() != 32)
    return emitOpError("result element type must be i32");

  int64_t inputDim = inputType.getDimSize(0);
  int64_t weightDim = weightType.getDimSize(0);
  if (inputDim != ShapedType::kDynamic &&
      weightDim != ShapedType::kDynamic && inputDim != weightDim) {
    return emitOpError("input and weight dimensions must match");
  }

  if (failed(verifySignedI8ZeroPointFits(getOperation(), "input_zero_point",
                                         getInputZeroPoint())))
    return failure();

  return verifySignedI8ZeroPointFits(getOperation(), "weight_zero_point",
                                     getWeightZeroPoint());
}

//===----------------------------------------------------------------------===//
// DotW4A8Op
//===----------------------------------------------------------------------===//

LogicalResult DotW4A8Op::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto packedWeightType =
      dyn_cast<RankedTensorType>(getPackedWeight().getType());
  auto weightScalesType =
      dyn_cast<RankedTensorType>(getWeightScales().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputType || !packedWeightType || !weightScalesType || !resultType)
    return emitOpError("operands and result must be ranked tensors");

  if (inputType.getRank() != 1)
    return emitOpError("input must be a 1D tensor");

  if (packedWeightType.getRank() != 1)
    return emitOpError("packed_weight must be a 1D tensor");

  if (weightScalesType.getRank() != 1)
    return emitOpError("weight_scales must be a 1D tensor");

  if (failed(
          verifySignlessI8ElementType(getOperation(), inputType, "input")))
    return failure();

  auto packedWeightElementType =
      dyn_cast<IntegerType>(packedWeightType.getElementType());
  if (!packedWeightElementType || !packedWeightElementType.isSignless() ||
      packedWeightElementType.getWidth() != 8) {
    return emitOpError("packed_weight element type must be signless i8");
  }

  if (!weightScalesType.getElementType().isF32())
    return emitOpError("weight_scales element type must be f32");

  if (resultType.getRank() != 0)
    return emitOpError("result must be a rank-0 tensor");

  if (!resultType.getElementType().isF32())
    return emitOpError("result element type must be f32");

  int64_t groupSize = getGroupSize();
  if (groupSize <= 0)
    return emitOpError("group_size must be positive");

  int64_t inputDim = inputType.getDimSize(0);
  int64_t packedWeightDim = packedWeightType.getDimSize(0);
  int64_t weightScalesDim = weightScalesType.getDimSize(0);
  if (inputDim != ShapedType::kDynamic) {
    int64_t expectedPackedWeightDim = (inputDim + 1) / 2;
    if (packedWeightDim != ShapedType::kDynamic &&
        packedWeightDim != expectedPackedWeightDim) {
      return emitOpError(
          "packed_weight dimension must equal ceil(input dimension / 2)");
    }

    int64_t expectedWeightScalesDim = (inputDim + groupSize - 1) / groupSize;
    if (weightScalesDim != ShapedType::kDynamic &&
        weightScalesDim != expectedWeightScalesDim) {
      return emitOpError("weight_scales dimension must equal ceil(input "
                         "dimension / group_size)");
    }
  }

  if (failed(verifyPositiveFiniteFloatAttr(getOperation(), getInputScaleAttr(),
                                           "input_scale")))
    return failure();

  if (failed(verifySignedI8ZeroPointFits(getOperation(), "input_zero_point",
                                         getInputZeroPoint())))
    return failure();

  return verifySignedI4ZeroPointFits(getOperation(), "weight_zero_point",
                                     getWeightZeroPoint());
}

//===----------------------------------------------------------------------===//
// MatvecW4A8Op
//===----------------------------------------------------------------------===//

LogicalResult MatvecW4A8Op::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto packedWeightsType =
      dyn_cast<RankedTensorType>(getPackedWeights().getType());
  auto weightScalesType =
      dyn_cast<RankedTensorType>(getWeightScales().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputType || !packedWeightsType || !weightScalesType || !resultType)
    return emitOpError("operands and result must be ranked tensors");

  if (inputType.getRank() != 1)
    return emitOpError("input must be a 1D tensor");

  if (packedWeightsType.getRank() != 2)
    return emitOpError("packed_weights must be a 2D tensor");

  if (weightScalesType.getRank() != 2)
    return emitOpError("weight_scales must be a 2D tensor");

  if (failed(
          verifySignlessI8ElementType(getOperation(), inputType, "input")))
    return failure();

  if (failed(verifySignlessI8ElementType(getOperation(), packedWeightsType,
                                         "packed_weights")))
    return failure();

  if (!weightScalesType.getElementType().isF32())
    return emitOpError("weight_scales element type must be f32");

  if (resultType.getRank() != 1)
    return emitOpError("result must be a 1D tensor");

  if (!resultType.getElementType().isF32())
    return emitOpError("result element type must be f32");

  int64_t groupSize = getGroupSize();
  if (groupSize <= 0)
    return emitOpError("group_size must be positive");

  int64_t inputCols = inputType.getDimSize(0);
  int64_t packedRows = packedWeightsType.getDimSize(0);
  int64_t packedCols = packedWeightsType.getDimSize(1);
  int64_t scaleRows = weightScalesType.getDimSize(0);
  int64_t scaleCols = weightScalesType.getDimSize(1);
  int64_t resultRows = resultType.getDimSize(0);

  if (inputCols != ShapedType::kDynamic) {
    int64_t expectedPackedCols = (inputCols + 1) / 2;
    if (packedCols != ShapedType::kDynamic &&
        packedCols != expectedPackedCols) {
      return emitOpError("packed_weights column dimension must equal "
                         "ceil(input dimension / 2)");
    }

    int64_t expectedScaleCols = (inputCols + groupSize - 1) / groupSize;
    if (scaleCols != ShapedType::kDynamic &&
        scaleCols != expectedScaleCols) {
      return emitOpError("weight_scales column dimension must equal "
                         "ceil(input dimension / group_size)");
    }
  }

  if (scaleRows != ShapedType::kDynamic &&
      packedRows != ShapedType::kDynamic && scaleRows != packedRows) {
    return emitOpError(
        "weight_scales row dimension must match packed_weights row dimension");
  }

  if (resultRows != ShapedType::kDynamic &&
      packedRows != ShapedType::kDynamic && resultRows != packedRows) {
    return emitOpError(
        "result dimension must match packed_weights row dimension");
  }

  if (failed(verifyPositiveFiniteFloatAttr(getOperation(), getInputScaleAttr(),
                                           "input_scale")))
    return failure();

  if (failed(verifySignedI8ZeroPointFits(getOperation(), "input_zero_point",
                                         getInputZeroPoint())))
    return failure();

  return verifySignedI4ZeroPointFits(getOperation(), "weight_zero_point",
                                     getWeightZeroPoint());
}

//===----------------------------------------------------------------------===//
// MatvecI8Op
//===----------------------------------------------------------------------===//

LogicalResult MatvecI8Op::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightsType = dyn_cast<RankedTensorType>(getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputType || !weightsType || !resultType)
    return emitOpError("operands and result must be ranked tensors");

  if (inputType.getRank() != 1)
    return emitOpError("input must be a 1D tensor");

  if (weightsType.getRank() != 2)
    return emitOpError("weights must be a 2D tensor");

  if (failed(
          verifySignlessI8ElementType(getOperation(), inputType, "input")))
    return failure();

  if (failed(
          verifySignlessI8ElementType(getOperation(), weightsType, "weights")))
    return failure();

  if (resultType.getRank() != 1)
    return emitOpError("result must be a 1D tensor");

  auto resultElementType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultElementType || resultElementType.getWidth() != 32)
    return emitOpError("result element type must be i32");

  int64_t inputCols = inputType.getDimSize(0);
  int64_t weightsRows = weightsType.getDimSize(0);
  int64_t weightsCols = weightsType.getDimSize(1);
  int64_t resultRows = resultType.getDimSize(0);

  if (inputCols != ShapedType::kDynamic &&
      weightsCols != ShapedType::kDynamic && inputCols != weightsCols) {
    return emitOpError(
        "input dimension must match weights column dimension");
  }

  if (resultRows != ShapedType::kDynamic &&
      weightsRows != ShapedType::kDynamic && resultRows != weightsRows) {
    return emitOpError("result dimension must match weights row dimension");
  }

  if (failed(verifySignedI8ZeroPointFits(getOperation(), "input_zero_point",
                                         getInputZeroPoint())))
    return failure();

  return verifySignedI8ZeroPointFits(getOperation(), "weight_zero_point",
                                     getWeightZeroPoint());
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//

LogicalResult RMSNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());

  if (!inputType || !weightType)
    return emitOpError("input and weight must be ranked tensors");

  if (!outputType)
    return emitOpError("output must be a ranked tensor");

  if (inputType.getRank() < 1)
    return emitOpError("input must have at least 1 dimension");

  if (failed(
          verifyFloatingPointElementType(getOperation(), inputType, "input")))
    return failure();

  if (failed(verifyMatchingElementTypes(
          getOperation(), {inputType, weightType, outputType},
          "input, weight, and output")))
    return failure();

  if (!compatibleShapes(inputType, outputType))
    return emitOpError("output shape must match input shape");

  if (failed(verifyPositiveFiniteFloatAttr(getOperation(), getEpsAttr(),
                                           "eps")))
    return failure();

  if (weightType.getRank() != 1)
    return emitOpError("weight must be a 1D tensor");

  int64_t inputLastDim = inputType.getDimSize(inputType.getRank() - 1);
  int64_t weightDim = weightType.getDimSize(0);
  if (inputLastDim != ShapedType::kDynamic &&
      weightDim != ShapedType::kDynamic && inputLastDim != weightDim) {
    return emitOpError("weight dimension must match input last dimension");
  }

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
