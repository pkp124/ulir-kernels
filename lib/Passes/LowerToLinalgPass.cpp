//===----------------------------------------------------------------------===//
// KSLowerToLinalgPass — Lower ks.matmul to linalg.matmul
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
//   ks.dot_i8 / ks.matvec_i8
//   ──►
//   linalg.generic with i8 -> i32 extension, zero-point subtraction, and i32
//   accumulation.
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
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

/// Build a zero-filled tensor of the given result type with explicit dynamic
/// sizes.
static Value createZeroFilledTensor(OpBuilder &b, Location loc,
                                    RankedTensorType resultType,
                                    ValueRange dynamicSizes) {
  Value empty = b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));
  return b.create<linalg::FillOp>(loc, zero, empty).getResult(0);
}

static Value createI32Constant(OpBuilder &b, Location loc, int64_t value) {
  return b.create<arith::ConstantIntOp>(loc, value, 32);
}

static Value createAdjustedI8Product(OpBuilder &b, Location loc, Value input,
                                     Value weight, Value inputZeroPoint,
                                     Value weightZeroPoint) {
  Type i32Type = b.getI32Type();
  Value inputI32 = b.create<arith::ExtSIOp>(loc, i32Type, input);
  Value weightI32 = b.create<arith::ExtSIOp>(loc, i32Type, weight);
  Value adjustedInput =
      b.create<arith::SubIOp>(loc, inputI32, inputZeroPoint);
  Value adjustedWeight =
      b.create<arith::SubIOp>(loc, weightI32, weightZeroPoint);
  return b.create<arith::MulIOp>(loc, adjustedInput, adjustedWeight);
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

/// Lower ks.dot_i8 to linalg.generic with i32 accumulation.
struct DotI8ToLinalgPattern : public OpRewritePattern<ks::DotI8Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::DotI8Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());

    Value filled =
        createZeroFilledTensor(rewriter, loc, resultType, ValueRange{});
    Value inputZeroPoint =
        createI32Constant(rewriter, loc, op.getInputZeroPoint());
    Value weightZeroPoint =
        createI32Constant(rewriter, loc, op.getWeightZeroPoint());

    MLIRContext *context = rewriter.getContext();
    AffineExpr k = rewriter.getAffineDimExpr(0);
    SmallVector<AffineMap> maps{
        AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {k}, context),
        AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {k}, context),
        AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {}, context)};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::reduction};

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getInput(), op.getWeight()},
        ValueRange{filled}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value product = createAdjustedI8Product(
              b, loc, args[0], args[1], inputZeroPoint, weightZeroPoint);
          Value accumulated = b.create<arith::AddIOp>(loc, args[2], product);
          b.create<linalg::YieldOp>(loc, accumulated);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// Lower ks.matvec_i8 to linalg.generic with i32 accumulation.
struct MatvecI8ToLinalgPattern : public OpRewritePattern<ks::MatvecI8Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MatvecI8Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());

    SmallVector<Value> dynamicSizes;
    if (resultType.isDynamicDim(0))
      dynamicSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, op.getWeights(), 0));
    Value filled =
        createZeroFilledTensor(rewriter, loc, resultType, dynamicSizes);
    Value inputZeroPoint =
        createI32Constant(rewriter, loc, op.getInputZeroPoint());
    Value weightZeroPoint =
        createI32Constant(rewriter, loc, op.getWeightZeroPoint());

    MLIRContext *context = rewriter.getContext();
    AffineExpr row = rewriter.getAffineDimExpr(0);
    AffineExpr col = rewriter.getAffineDimExpr(1);
    SmallVector<AffineMap> maps{
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {col}, context),
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {row, col},
                       context),
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {row}, context)};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
                                           utils::IteratorType::reduction};

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{op.getInput(), op.getWeights()},
        ValueRange{filled}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value product = createAdjustedI8Product(
              b, loc, args[0], args[1], inputZeroPoint, weightZeroPoint);
          Value accumulated = b.create<arith::AddIOp>(loc, args[2], product);
          b.create<linalg::YieldOp>(loc, accumulated);
        });

    rewriter.replaceOp(op, generic.getResults());
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
    patterns.add<DotI8ToLinalgPattern, MatmulToLinalgPattern,
                 MatvecI8ToLinalgPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace kernelsmith
