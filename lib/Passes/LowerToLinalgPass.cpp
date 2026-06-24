//===----------------------------------------------------------------------===//
// KSLowerToLinalgPass — Lower KS structured ops to linalg
//
// Design: DES-008 (docs/design/DES-008-m3-linalg-matmul-lowering.md)
//
// Lowering rule:
//   ks.matmul %A, %B : tensor<MxKxT>, tensor<KxNxT> -> tensor<MxNxT>
//   ──►
//   %empty  = tensor.empty() : tensor<MxNxT>
//   %zero   = arith.constant 0 : T
//   %filled = linalg.fill ins(%zero) outs(%empty) -> tensor<MxNxT>
//   %C      = linalg.matmul ins(%A, %B) outs(%filled) -> tensor<MxNxT>
//
// The zero-fill is required because linalg.matmul accumulates (C += A*B),
// while ks.matmul has pure-replacement semantics (C = A*B).
//
//   ks.dot_i8 %input, %weight : tensor<Kxi8>, tensor<Kxi8> -> tensor<i32>
//   ──►
//   linalg.generic reduction with signed i32 accumulation:
//     acc += (extsi(input[k]) - input_zero_point) *
//            (extsi(weight[k]) - weight_zero_point)
//
//   ks.rms_norm %input, %weight {eps} : tensor<...xDxT>, tensor<DxT> -> ...
//   ──►
//   Three linalg.generic ops, normalizing over the trailing dimension D:
//     1. reduction: ssq[outer] = sum_D(input * input)  (f32 accumulation)
//     2. elementwise (rank-1 reduced): scale[outer] =
//          rsqrt(ssq[outer] / D + eps)
//     3. elementwise (full rank): output[..., d] =
//          input[..., d] * scale[outer] * weight[d]
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace kernelsmith {

// Pull in the generated pass base class.
#define GEN_PASS_DEF_KSLOWERTOLINALGPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Build a zero-filled tensor of the given result type.
///
/// For dynamic dimensions: M comes from lhs dim-0, N comes from rhs dim-1.
/// This covers the only two dynamic dimensions possible in a 2-D matmul result.
static Value createZeroFilledTensor(OpBuilder &b, Location loc,
                                    RankedTensorType resultType, Value lhs,
                                    Value rhs) {
  // Collect runtime values for dynamic dimensions in row-major order.
  SmallVector<Value> dynamicSizes;
  if (resultType.isDynamicDim(0))
    dynamicSizes.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
  if (resultType.isDynamicDim(1))
    dynamicSizes.push_back(b.create<tensor::DimOp>(loc, rhs, 1));

  Value empty = b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);

  // Zero constant matching the element type (float or integer).
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));

  return b.create<linalg::FillOp>(loc, zero, empty).getResult(0);
}

/// Build a zero-filled rank-0 tensor for scalar reductions.
static Value createZeroFilledScalarTensor(OpBuilder &b, Location loc,
                                          RankedTensorType resultType) {
  SmallVector<Value> dynamicSizes;
  Value empty = b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));
  return b.create<linalg::FillOp>(loc, zero, empty).getResult(0);
}

/// Build a zero-filled tensor with dynamic dimensions sourced from `shapeLike`.
static Value createZeroFilledTensorLike(OpBuilder &b, Location loc,
                                        RankedTensorType resultType,
                                        Value shapeLike) {
  SmallVector<Value> dynamicSizes;
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, shapeLike, dim));
  }

  Value empty = b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));
  return b.create<linalg::FillOp>(loc, zero, empty).getResult(0);
}

/// Build an uninitialized tensor of `resultType`, sourcing dynamic dimensions
/// from the matching leading dimensions of `shapeLike`. Used when every result
/// element is written (no accumulation), so no zero-fill is needed.
static Value createEmptyTensorLike(OpBuilder &b, Location loc,
                                   RankedTensorType resultType,
                                   Value shapeLike) {
  SmallVector<Value> dynamicSizes;
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim))
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, shapeLike, dim));
  }
  return b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);
}

static Value createEmptyElementwiseTensor(OpBuilder &b, Location loc,
                                          RankedTensorType resultType,
                                          Value lhs, Value rhs) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  int64_t resultRank = resultType.getRank();
  SmallVector<Value> dynamicSizes;

  auto getAlignedDimIndex = [resultRank](RankedTensorType type,
                                         int64_t resultDim) -> int64_t {
    int64_t offset = resultRank - type.getRank();
    return resultDim < offset ? -1 : resultDim - offset;
  };

  for (int64_t dim = 0; dim < resultRank; ++dim) {
    if (!resultType.isDynamicDim(dim))
      continue;

    int64_t lhsDim = getAlignedDimIndex(lhsType, dim);
    if (lhsDim >= 0 && lhsType.getDimSize(lhsDim) != 1) {
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, lhs, lhsDim));
      continue;
    }

    int64_t rhsDim = getAlignedDimIndex(rhsType, dim);
    if (rhsDim >= 0 && rhsType.getDimSize(rhsDim) != 1) {
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, rhs, rhsDim));
      continue;
    }

    dynamicSizes.push_back(b.create<arith::ConstantIndexOp>(loc, 1));
  }

  return b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);
}

static AffineMap getBroadcastIndexingMap(OpBuilder &b,
                                         RankedTensorType operandType,
                                         RankedTensorType resultType) {
  int64_t resultRank = resultType.getRank();
  int64_t offset = resultRank - operandType.getRank();
  SmallVector<AffineExpr> exprs;
  exprs.reserve(operandType.getRank());

  for (int64_t dim = 0; dim < operandType.getRank(); ++dim) {
    if (operandType.getDimSize(dim) == 1) {
      exprs.push_back(b.getAffineConstantExpr(0));
      continue;
    }
    exprs.push_back(b.getAffineDimExpr(offset + dim));
  }

  return AffineMap::get(resultRank, /*symbolCount=*/0, exprs,
                        b.getContext());
}

static Value buildI8ProductAdd(OpBuilder &b, Location loc, Value input,
                               Value weight, Value accumulator,
                               int64_t inputZeroPoint,
                               int64_t weightZeroPoint) {
  Type i32Type = b.getI32Type();
  Value inputI32 = b.create<arith::ExtSIOp>(loc, i32Type, input);
  Value inputZp = b.create<arith::ConstantOp>(
      loc, b.getI32IntegerAttr(inputZeroPoint));
  Value centeredInput = b.create<arith::SubIOp>(loc, inputI32, inputZp);

  Value weightI32 = b.create<arith::ExtSIOp>(loc, i32Type, weight);
  Value weightZp = b.create<arith::ConstantOp>(
      loc, b.getI32IntegerAttr(weightZeroPoint));
  Value centeredWeight = b.create<arith::SubIOp>(loc, weightI32, weightZp);

  Value product = b.create<arith::MulIOp>(loc, centeredInput, centeredWeight);
  return b.create<arith::AddIOp>(loc, accumulator, product);
}

static Value buildW4A8ProductAdd(OpBuilder &b, Location loc, Value input,
                                 Value packedWeights, Value weightScales,
                                 Value accumulator, int64_t groupSize,
                                 double inputScale, int64_t inputZeroPoint,
                                 int64_t weightZeroPoint, int64_t colIndexDim,
                                 Value rowIndex = Value()) {
  Type i32Type = b.getI32Type();
  Type f32Type = b.getF32Type();

  Value two = b.create<arith::ConstantIndexOp>(loc, 2);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value groupSizeValue = b.create<arith::ConstantIndexOp>(loc, groupSize);
  Value highShift =
      b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(4));
  Value signShift =
      b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(28));
  Value lowMask =
      b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(0x0f));

  Value k = b.create<linalg::IndexOp>(loc, colIndexDim);
  Value packedIndex = b.create<arith::DivUIOp>(loc, k, two);
  SmallVector<Value> packedIndices;
  if (rowIndex)
    packedIndices.push_back(rowIndex);
  packedIndices.push_back(packedIndex);
  Value packed = b.create<tensor::ExtractOp>(loc, packedWeights,
                                             packedIndices);
  Value packedI32 = b.create<arith::ExtUIOp>(loc, i32Type, packed);

  Value lowNibble = b.create<arith::AndIOp>(loc, packedI32, lowMask);
  Value highNibble = b.create<arith::ShRUIOp>(loc, packedI32, highShift);
  Value remainder = b.create<arith::RemUIOp>(loc, k, two);
  Value isHighNibble = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, remainder, one);
  Value nibble =
      b.create<arith::SelectOp>(loc, isHighNibble, highNibble, lowNibble);

  Value shiftedNibble = b.create<arith::ShLIOp>(loc, nibble, signShift);
  Value signedWeight =
      b.create<arith::ShRSIOp>(loc, shiftedNibble, signShift);
  Value weightZp = b.create<arith::ConstantOp>(
      loc, b.getI32IntegerAttr(weightZeroPoint));
  Value centeredWeight = b.create<arith::SubIOp>(loc, signedWeight, weightZp);

  Value inputI32 = b.create<arith::ExtSIOp>(loc, i32Type, input);
  Value inputZp = b.create<arith::ConstantOp>(
      loc, b.getI32IntegerAttr(inputZeroPoint));
  Value centeredInput = b.create<arith::SubIOp>(loc, inputI32, inputZp);

  Value inputF32 = b.create<arith::SIToFPOp>(loc, f32Type, centeredInput);
  Value inputScaleValue = b.create<arith::ConstantOp>(
      loc, b.getF32FloatAttr(static_cast<float>(inputScale)));
  Value scaledInput =
      b.create<arith::MulFOp>(loc, inputF32, inputScaleValue);

  Value scaleIndex = b.create<arith::DivUIOp>(loc, k, groupSizeValue);
  SmallVector<Value> scaleIndices;
  if (rowIndex)
    scaleIndices.push_back(rowIndex);
  scaleIndices.push_back(scaleIndex);
  Value weightScale = b.create<tensor::ExtractOp>(loc, weightScales,
                                                  scaleIndices);
  Value weightF32 = b.create<arith::SIToFPOp>(loc, f32Type, centeredWeight);
  Value scaledWeight =
      b.create<arith::MulFOp>(loc, weightF32, weightScale);

  Value product = b.create<arith::MulFOp>(loc, scaledInput, scaledWeight);
  return b.create<arith::AddFOp>(loc, accumulator, product);
}

//===----------------------------------------------------------------------===//
// Lowering pattern
//===----------------------------------------------------------------------===//

/// Lower ks.matmul to linalg.matmul with a zero-initialized accumulator.
///
/// Uses OpRewritePattern (greedy driver) — same infrastructure as
/// LowerActivationsPass. The named linalg.matmul op is chosen over
/// linalg.generic because:
///   1. It is directly tileable by KSTilePass via TilingInterface.
///   2. Future packing (M4) relies on linalg::packMatmulGreedily which
///      pattern-matches linalg.matmul by name.
///   3. Vectorization transforms also prefer named ops.
struct MatmulToLinalgPattern : public OpRewritePattern<ks::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());

    Value filled = createZeroFilledTensor(rewriter, loc, resultType,
                                          op.getLhs(), op.getRhs());

    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, TypeRange{resultType}, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{filled});
    return success();
  }
};

struct AddToLinalgPattern : public OpRewritePattern<ks::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::AddOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());

    SmallVector<AffineMap> maps{
        getBroadcastIndexingMap(rewriter, lhsType, resultType),
        getBroadcastIndexingMap(rewriter, rhsType, resultType),
        rewriter.getMultiDimIdentityMap(resultType.getRank())};
    SmallVector<utils::IteratorType> iters(resultType.getRank(),
                                           utils::IteratorType::parallel);
    Value empty = createEmptyElementwiseTensor(
        rewriter, loc, resultType, op.getLhs(), op.getRhs());

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{empty}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct MulToLinalgPattern : public OpRewritePattern<ks::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());

    SmallVector<AffineMap> maps{
        getBroadcastIndexingMap(rewriter, lhsType, resultType),
        getBroadcastIndexingMap(rewriter, rhsType, resultType),
        rewriter.getMultiDimIdentityMap(resultType.getRank())};
    SmallVector<utils::IteratorType> iters(resultType.getRank(),
                                           utils::IteratorType::parallel);
    Value empty = createEmptyElementwiseTensor(
        rewriter, loc, resultType, op.getLhs(), op.getRhs());

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{empty}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value product = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, product);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower ks.dot_i8 to a linalg.generic reduction with i32 accumulation.
struct DotI8ToLinalgPattern : public OpRewritePattern<ks::DotI8Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::DotI8Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());

    AffineExpr k = rewriter.getAffineDimExpr(0);
    AffineMap vectorMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         k, context);
    AffineMap scalarMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         context);
    SmallVector<AffineMap> maps{vectorMap, vectorMap, scalarMap};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::reduction};

    Value filled = createZeroFilledScalarTensor(rewriter, loc, resultType);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getInput(), op.getWeight()},
        ValueRange{filled}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = buildI8ProductAdd(b, loc, args[0], args[1], args[2],
                                        op.getInputZeroPoint(),
                                        op.getWeightZeroPoint());
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower ks.dot_w4a8 to a fused linalg.generic f32 reduction.
struct DotW4A8ToLinalgPattern : public OpRewritePattern<ks::DotW4A8Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::DotW4A8Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());

    AffineExpr k = rewriter.getAffineDimExpr(0);
    AffineMap vectorMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         k, context);
    AffineMap scalarMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         context);
    SmallVector<AffineMap> maps{vectorMap, scalarMap};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::reduction};

    Value filled = createZeroFilledScalarTensor(rewriter, loc, resultType);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getInput()}, ValueRange{filled}, maps,
        iters, [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = buildW4A8ProductAdd(
              b, loc, args[0], op.getPackedWeight(), op.getWeightScales(),
              args[1], op.getGroupSize(),
              op.getInputScaleAttr().getValueAsDouble(),
              op.getInputZeroPoint(), op.getWeightZeroPoint(),
              /*colIndexDim=*/0);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower ks.matvec_w4a8 to a fused row-parallel linalg.generic reduction.
struct MatvecW4A8ToLinalgPattern
    : public OpRewritePattern<ks::MatvecW4A8Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MatvecW4A8Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());

    AffineExpr row = rewriter.getAffineDimExpr(0);
    AffineExpr col = rewriter.getAffineDimExpr(1);
    AffineMap inputMap = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                        col, context);
    AffineMap resultMap = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                         row, context);
    SmallVector<AffineMap> maps{inputMap, resultMap};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
                                           utils::IteratorType::reduction};

    Value filled = createZeroFilledTensorLike(rewriter, loc, resultType,
                                              op.getPackedWeights());

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getInput()}, ValueRange{filled}, maps,
        iters, [&](OpBuilder &b, Location loc, ValueRange args) {
          Value rowIndex = b.create<linalg::IndexOp>(loc, 0);
          Value sum = buildW4A8ProductAdd(
              b, loc, args[0], op.getPackedWeights(), op.getWeightScales(),
              args[1], op.getGroupSize(),
              op.getInputScaleAttr().getValueAsDouble(),
              op.getInputZeroPoint(), op.getWeightZeroPoint(),
              /*colIndexDim=*/1, rowIndex);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower ks.matvec_i8 to a linalg.generic row-parallel reduction.
struct MatvecI8ToLinalgPattern : public OpRewritePattern<ks::MatvecI8Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MatvecI8Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    auto resultType = cast<RankedTensorType>(op.getType());

    AffineExpr row = rewriter.getAffineDimExpr(0);
    AffineExpr col = rewriter.getAffineDimExpr(1);
    AffineMap inputMap = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                        col, context);
    AffineMap weightsMap = AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {row, col}, context);
    AffineMap resultMap = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                         row, context);
    SmallVector<AffineMap> maps{inputMap, weightsMap, resultMap};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
                                           utils::IteratorType::reduction};

    Value filled =
        createZeroFilledTensorLike(rewriter, loc, resultType, op.getWeights());

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getInput(), op.getWeights()},
        ValueRange{filled}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = buildI8ProductAdd(b, loc, args[0], args[1], args[2],
                                        op.getInputZeroPoint(),
                                        op.getWeightZeroPoint());
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower ks.rms_norm to three linalg.generic ops normalizing over the trailing
/// dimension. The mean-of-squares reduction accumulates in the (floating-point)
/// element type, and epsilon is added before the reciprocal square root,
/// matching the ks_rms_norm_f32 reference and the golden descriptors.
struct RMSNormToLinalgPattern : public OpRewritePattern<ks::RMSNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::RMSNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Value input = op.getInput();
    Value weight = op.getWeight();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elementType = inputType.getElementType();
    int64_t rank = inputType.getRank();
    int64_t lastDim = rank - 1;

    // Reduced tensor shape = input shape with the trailing dimension dropped.
    auto reducedType = RankedTensorType::get(
        inputType.getShape().drop_back(), elementType);

    // Identity map over all `rank` dims and a map that drops the trailing
    // (reduction) dim, projecting onto the reduced/"outer" tensor.
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineExpr> outerExprs;
    outerExprs.reserve(rank - 1);
    for (int64_t dim = 0; dim < rank - 1; ++dim)
      outerExprs.push_back(rewriter.getAffineDimExpr(dim));
    AffineMap outerMap =
        AffineMap::get(rank, /*symbolCount=*/0, outerExprs, context);
    AffineMap reducedIdentityMap =
        rewriter.getMultiDimIdentityMap(rank - 1);

    // 1. Sum of squares reduction: ssq[outer] = sum_lastDim(input * input).
    SmallVector<utils::IteratorType> reduceIters(rank,
                                                 utils::IteratorType::parallel);
    reduceIters[lastDim] = utils::IteratorType::reduction;
    Value ssqInit =
        createZeroFilledTensorLike(rewriter, loc, reducedType, input);
    auto ssqOp = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{input}, ValueRange{ssqInit},
        SmallVector<AffineMap>{identityMap, outerMap}, reduceIters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value square = b.create<arith::MulFOp>(loc, args[0], args[0]);
          Value sum = b.create<arith::AddFOp>(loc, args[1], square);
          b.create<linalg::YieldOp>(loc, sum);
        });

    // Number of normalized elements (the trailing dim) as a float scalar.
    Value innerCount;
    if (inputType.isDynamicDim(lastDim)) {
      Value dim = rewriter.create<tensor::DimOp>(loc, input, lastDim);
      Value dimI64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), dim);
      innerCount = rewriter.create<arith::SIToFPOp>(loc, elementType, dimI64);
    } else {
      innerCount = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(
                   elementType,
                   static_cast<double>(inputType.getDimSize(lastDim))));
    }
    Value epsValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType,
                                   op.getEpsAttr().getValueAsDouble()));

    // 2. scale[outer] = rsqrt(ssq[outer] / inner + eps).
    SmallVector<utils::IteratorType> reducedParallelIters(
        rank - 1, utils::IteratorType::parallel);
    Value scaleInit =
        createEmptyTensorLike(rewriter, loc, reducedType, input);
    auto scaleOp = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{ssqOp.getResult(0)},
        ValueRange{scaleInit},
        SmallVector<AffineMap>{reducedIdentityMap, reducedIdentityMap},
        reducedParallelIters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value mean = b.create<arith::DivFOp>(loc, args[0], innerCount);
          Value meanEps = b.create<arith::AddFOp>(loc, mean, epsValue);
          Value scale = b.create<math::RsqrtOp>(loc, meanEps);
          b.create<linalg::YieldOp>(loc, scale);
        });

    // 3. output[..., d] = input[..., d] * scale[outer] * weight[d].
    AffineMap weightMap = AffineMap::get(
        rank, /*symbolCount=*/0, rewriter.getAffineDimExpr(lastDim), context);
    SmallVector<utils::IteratorType> applyIters(rank,
                                                utils::IteratorType::parallel);
    Value outInit = createEmptyTensorLike(rewriter, loc, resultType, input);
    auto applyOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input, scaleOp.getResult(0), weight},
        ValueRange{outInit},
        SmallVector<AffineMap>{identityMap, outerMap, weightMap, identityMap},
        applyIters, [&](OpBuilder &b, Location loc, ValueRange args) {
          Value scaled = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value result = b.create<arith::MulFOp>(loc, scaled, args[2]);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, applyOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSLowerToLinalgPass
    : impl::KSLowerToLinalgPassBase<KSLowerToLinalgPass> {
  using KSLowerToLinalgPassBase::KSLowerToLinalgPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddToLinalgPattern, DotI8ToLinalgPattern,
                 DotW4A8ToLinalgPattern, MatmulToLinalgPattern,
                 MatvecW4A8ToLinalgPattern, MulToLinalgPattern,
                 MatvecI8ToLinalgPattern, RMSNormToLinalgPattern>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace kernelsmith
