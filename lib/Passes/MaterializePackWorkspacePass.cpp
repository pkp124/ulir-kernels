//===----------------------------------------------------------------------===//
// KSMaterializePackWorkspacePass
//
// Materialize tensor-level linalg.pack ops into explicit caller-provided
// workspace buffers before one-shot bufferization. This keeps KernelSmith's
// packed GEMM path compatible with the no-internal-allocation runtime model.
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSMATERIALIZEPACKWORKSPACEPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

static bool isWorkspaceType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 1)
    return false;

  auto intType = dyn_cast<IntegerType>(memrefType.getElementType());
  return intType && intType.getWidth() == 8;
}

static FailureOr<int64_t> getElementByteWidth(Type type, Operation *op) {
  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(type))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(type))
    bitWidth = floatType.getWidth();
  else
    return op->emitError("unsupported packed element type");

  if (bitWidth % 8 != 0)
    return op->emitError("packed element type must have byte-sized width");
  return bitWidth / 8;
}

static int64_t alignTo(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

static LogicalResult validateSupportedPack(linalg::PackOp packOp,
                                           RankedTensorType sourceType,
                                           RankedTensorType packedType) {
  if (!sourceType.hasStaticShape() || !packedType.hasStaticShape())
    return packOp->emitError(
        "dynamic pack workspace materialization is not supported");

  if (sourceType.getRank() != 2 || packedType.getRank() != 3)
    return packOp->emitError(
        "unsupported linalg.pack rank for KernelSmith matmul");

  if (packOp.getInnerDimsPos() != ArrayRef<int64_t>{1})
    return packOp->emitError(
        "unsupported linalg.pack layout for KernelSmith matmul");

  if (packOp.getOuterDimsPerm() != ArrayRef<int64_t>{1, 0})
    return packOp->emitError(
        "unsupported linalg.pack layout for KernelSmith matmul");

  ArrayRef<int64_t> tiles = packOp.getStaticTiles();
  if (tiles.size() != 1 || tiles[0] <= 0)
    return packOp->emitError(
        "unsupported linalg.pack tile sizes for KernelSmith matmul");

  int64_t K = sourceType.getDimSize(0);
  int64_t N = sourceType.getDimSize(1);
  int64_t packFactor = tiles[0];
  if (N % packFactor != 0)
    return packOp->emitError(
        "packed N dimension must be divisible by pack factor");

  ArrayRef<int64_t> packedShape = packedType.getShape();
  if (packedShape[0] != N / packFactor || packedShape[1] != K ||
      packedShape[2] != packFactor)
    return packOp->emitError(
        "unsupported linalg.pack layout for KernelSmith matmul");

  return success();
}

struct KSMaterializePackWorkspacePass
    : impl::KSMaterializePackWorkspacePassBase<
          KSMaterializePackWorkspacePass> {
  using KSMaterializePackWorkspacePassBase::
      KSMaterializePackWorkspacePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<linalg::PackOp> packOps;
    func.walk([&](linalg::PackOp packOp) {
      if (packOp->getNumResults() == 1)
        packOps.push_back(packOp);
    });

    if (packOps.empty())
      return;

    BlockArgument workspace;
    for (BlockArgument arg : func.getArguments()) {
      if (isWorkspaceType(arg.getType()))
        workspace = arg;
    }

    if (!workspace) {
      packOps.front()->emitError(
          "workspace argument required for materializing packed buffers");
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(func->getContext());
    int64_t nextWorkspaceOffset = 0;

    for (linalg::PackOp packOp : packOps) {
      if (packOp->getNumResults() != 1)
        continue;

      auto sourceType = dyn_cast<RankedTensorType>(packOp.getSource().getType());
      auto packedType =
          dyn_cast<RankedTensorType>(packOp->getResult(0).getType());
      if (!sourceType || !packedType) {
        packOp->emitError("expected ranked tensor linalg.pack operands");
        signalPassFailure();
        return;
      }

      if (failed(validateSupportedPack(packOp, sourceType, packedType))) {
        signalPassFailure();
        return;
      }

      FailureOr<int64_t> byteWidth =
          getElementByteWidth(packedType.getElementType(), packOp);
      if (failed(byteWidth)) {
        signalPassFailure();
        return;
      }

      int64_t allocationBytes = packedType.getNumElements() * *byteWidth;
      int64_t alignedOffset = alignTo(nextWorkspaceOffset, workspaceAlignment);
      nextWorkspaceOffset = alignedOffset + allocationBytes;

      Location loc = packOp.getLoc();
      rewriter.setInsertionPoint(packOp);

      auto offset = rewriter.create<arith::ConstantIndexOp>(loc, alignedOffset);
      auto packedMemrefType =
          MemRefType::get(packedType.getShape(), packedType.getElementType());
      Value packedBuffer = rewriter.create<memref::ViewOp>(
          loc, packedMemrefType, workspace, offset, ValueRange{});

      auto sourceMemrefType =
          MemRefType::get(sourceType.getShape(), sourceType.getElementType());
      Value sourceBuffer = rewriter.create<bufferization::ToBufferOp>(
          loc, sourceMemrefType, packOp.getSource());

      SmallVector<OpFoldResult> innerTiles{
          rewriter.getIndexAttr(packOp.getStaticTiles().front())};
      rewriter.create<linalg::PackOp>(loc, sourceBuffer, packedBuffer,
                                      /*innerDimsPos=*/ArrayRef<int64_t>{1},
                                      /*innerTiles=*/innerTiles,
                                      /*paddingValue=*/std::optional<Value>{},
                                      /*outerDimsPerm=*/ArrayRef<int64_t>{1, 0});

      auto packedTensor = rewriter.create<bufferization::ToTensorOp>(
          loc, packedBuffer, /*restrict=*/true, /*writable=*/false);
      rewriter.replaceOp(packOp, packedTensor.getResult());
    }
  }
};

} // namespace kernelsmith
