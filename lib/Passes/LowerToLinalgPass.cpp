//===----------------------------------------------------------------------===//
// KSLowerToLinalgPass — Lower ks.matmul to linalg.matmul
//===----------------------------------------------------------------------===//

#include "KernelSmith/Passes/Passes.h"

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSLOWERTOLINALGPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a tensor.empty matching the result shape, using `lhs` dim 0 for M
/// and `rhs` dim 1 for N when the result has dynamic dimensions.
static Value createResultTensor(OpBuilder &b, Location loc, Value lhs,
                                Value rhs, RankedTensorType resultType,
                                Type elemType) {
  SmallVector<Value> dynamicSizes;
  auto shape = resultType.getShape();
  for (int64_t i = 0; i < resultType.getRank(); i++) {
    if (ShapedType::isDynamic(shape[i])) {
      // M comes from lhs dim 0, N comes from rhs dim 1.
      Value src = (i == 0) ? lhs : rhs;
      int64_t dim = (i == 0) ? 0 : 1;
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, src, dim));
    }
  }
  return b.create<tensor::EmptyOp>(loc, shape, elemType, dynamicSizes);
}

/// Create a linalg.generic that element-wise truncates a float tensor
/// from a wider type to a narrower type.
static Value createTruncF(OpBuilder &b, Location loc, Value input,
                          RankedTensorType outputType) {
  Type outputElem = outputType.getElementType();
  int64_t rank = outputType.getRank();

  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < rank; i++) {
    if (outputType.isDynamicDim(i))
      dynamicSizes.push_back(b.create<tensor::DimOp>(loc, input, i));
  }
  Value empty = b.create<tensor::EmptyOp>(loc, outputType.getShape(),
                                          outputElem, dynamicSizes);

  auto id = b.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> maps(2, id);
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

  auto generic = b.create<linalg::GenericOp>(
      loc, outputType, /*inputs=*/input, /*outputs=*/empty, maps, iters,
      [&](OpBuilder &nb, Location loc, ValueRange args) {
        Value truncated =
            nb.create<arith::TruncFOp>(loc, outputElem, args[0]);
        nb.create<linalg::YieldOp>(loc, truncated);
      });

  return generic.getResult(0);
}

//===----------------------------------------------------------------------===//
// Lowering patterns
//===----------------------------------------------------------------------===//

/// Lower ks.matmul to linalg.matmul:
///   %empty = tensor.empty [M, N] : resultType
///   %zero  = arith.constant 0.0 : elemType
///   %fill  = linalg.fill ins(%zero) outs(%empty)
///   %result = linalg.matmul ins(%lhs, %rhs) outs(%fill)
///
/// With acc_type (mixed precision):
///   Accumulate in acc_type, then truncate result back to output type.
struct MatmulLowering : public OpRewritePattern<ks::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type resultElemType = resultType.getElementType();

    // Determine accumulator element type.
    Type accElemType = resultElemType;
    if (auto accTypeAttr = op.getAccTypeAttr())
      accElemType = accTypeAttr.getValue();

    bool needsTrunc = (accElemType != resultElemType);

    // Build the output tensor type (may use acc_type for accumulation).
    RankedTensorType accType = resultType;
    if (needsTrunc)
      accType = RankedTensorType::get(resultType.getShape(), accElemType);

    // Create empty output tensor.
    Value empty =
        createResultTensor(rewriter, loc, lhs, rhs, accType, accElemType);

    // Fill with zero.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(accElemType));
    Value filled =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);

    // Create linalg.matmul.
    Value matmulResult =
        rewriter
            .create<linalg::MatmulOp>(loc, TypeRange{accType},
                                      ValueRange{lhs, rhs},
                                      ValueRange{filled})
            .getResult(0);

    // If accumulator differs from result, truncate back.
    if (needsTrunc) {
      matmulResult = createTruncF(rewriter, loc, matmulResult, resultType);
    }

    rewriter.replaceOp(op, matmulResult);
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
    patterns.add<MatmulLowering>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace kernelsmith
