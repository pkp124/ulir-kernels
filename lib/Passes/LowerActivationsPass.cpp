//===----------------------------------------------------------------------===//
// KSLowerActivationsPass — Lower ks.relu/gelu/silu to linalg.generic
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
#define GEN_PASS_DEF_KSLOWERACTIVATIONSPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a tensor.empty with the same shape as `input`, extracting
/// dynamic dimension sizes as needed.
static Value createEmptyTensorLike(OpBuilder &b, Location loc, Value input) {
  auto tensorType = cast<RankedTensorType>(input.getType());
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < tensorType.getRank(); i++) {
    if (tensorType.isDynamicDim(i))
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, input, i));
  }
  return b.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                   tensorType.getElementType(), dynamicSizes);
}

/// Build indexing maps and iterator types for an element-wise unary op
/// with the given rank.
static void getElementwiseMaps(OpBuilder &b, int64_t rank,
                               SmallVectorImpl<AffineMap> &maps,
                               SmallVectorImpl<utils::IteratorType> &iters) {
  auto id = b.getMultiDimIdentityMap(rank);
  maps.assign(2, id);
  iters.assign(rank, utils::IteratorType::parallel);
}

//===----------------------------------------------------------------------===//
// Lowering patterns
//===----------------------------------------------------------------------===//

/// ks.relu %x -> arith.maximumf(x, 0)
struct ReluLowering : public OpRewritePattern<ks::ReLUOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::ReLUOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elemType = resultType.getElementType();

    Value empty = createEmptyTensorLike(rewriter, loc, input);
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));

    SmallVector<AffineMap> maps;
    SmallVector<utils::IteratorType> iters;
    getElementwiseMaps(rewriter, resultType.getRank(), maps, iters);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, /*inputs=*/input, /*outputs=*/empty, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MaximumFOp>(loc, args[0], zero);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// ks.gelu %x -> 0.5 * x * (1 + erf(x / sqrt(2)))
struct GeluLowering : public OpRewritePattern<ks::GELUOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::GELUOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elemType = resultType.getElementType();

    Value empty = createEmptyTensorLike(rewriter, loc, input);

    auto floatType = cast<FloatType>(elemType);
    Value cHalf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 0.5));
    Value cOne = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 1.0));
    Value cSqrt1_2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 0.7071067811865476));

    SmallVector<AffineMap> maps;
    SmallVector<utils::IteratorType> iters;
    getElementwiseMaps(rewriter, resultType.getRank(), maps, iters);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, /*inputs=*/input, /*outputs=*/empty, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          // scaled = x * (1/sqrt(2))
          Value scaled = b.create<arith::MulFOp>(loc, x, cSqrt1_2);
          // erf_val = erf(scaled)
          Value erfVal = b.create<math::ErfOp>(loc, scaled);
          // one_plus_erf = 1.0 + erf_val
          Value onePlusErf = b.create<arith::AddFOp>(loc, cOne, erfVal);
          // half_x = 0.5 * x
          Value halfX = b.create<arith::MulFOp>(loc, cHalf, x);
          // result = half_x * one_plus_erf
          Value result = b.create<arith::MulFOp>(loc, halfX, onePlusErf);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// ks.silu %x -> x / (1 + exp(-x))
struct SiluLowering : public OpRewritePattern<ks::SiLUOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::SiLUOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto resultType = cast<RankedTensorType>(op.getType());
    Type elemType = resultType.getElementType();

    Value empty = createEmptyTensorLike(rewriter, loc, input);

    auto floatType = cast<FloatType>(elemType);
    Value cOne = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(floatType, 1.0));

    SmallVector<AffineMap> maps;
    SmallVector<utils::IteratorType> iters;
    getElementwiseMaps(rewriter, resultType.getRank(), maps, iters);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, /*inputs=*/input, /*outputs=*/empty, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          // neg_x = -x
          Value negX = b.create<arith::NegFOp>(loc, x);
          // exp_neg_x = exp(-x)
          Value expNegX = b.create<math::ExpOp>(loc, negX);
          // denom = 1 + exp(-x)
          Value denom = b.create<arith::AddFOp>(loc, cOne, expNegX);
          // result = x / denom
          Value result = b.create<arith::DivFOp>(loc, x, denom);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSLowerActivationsPass
    : impl::KSLowerActivationsPassBase<KSLowerActivationsPass> {
  using KSLowerActivationsPassBase::KSLowerActivationsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReluLowering, GeluLowering, SiluLowering>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace kernelsmith
