#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

namespace {

struct FoldBatchNormPattern : public RewritePattern {
  FoldBatchNormPattern(MLIRContext *context)
      : RewritePattern("test.batch_norm", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 5) return failure();

    Value bnInput = op->getOperand(0);
    Operation *convOp = bnInput.getDefiningOp();
    if (!convOp || convOp->getName().getStringRef() != "linalg.conv_2d_nhwc_fhwc")
      return failure();

    Location loc = op->getLoc();
    
    // B. Extraction
    Value weights = convOp->getOperand(1);
    Value gamma   = op->getOperand(1);
    Value beta    = op->getOperand(2);
    Value mean    = op->getOperand(3);
    Value var     = op->getOperand(4);

    auto weightType = weights.getType().cast<RankedTensorType>();
    auto paramType  = gamma.getType().cast<RankedTensorType>();
    auto outType    = bnInput.getType().cast<RankedTensorType>();
    auto elementType = paramType.getElementType();

    // CRITICAL FIX: Tell the rewriter to start inserting code BEFORE the Convolution.
    // This ensures the new weights exist before the Conv tries to use them.
    rewriter.setInsertionPoint(convOp);

    // 1. Calculate Multiplier
    Value eps = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elementType, 1e-5));
    Value empty1D = rewriter.create<tensor::EmptyOp>(loc, paramType.getShape(), elementType);
    auto map1D = rewriter.getMultiDimIdentityMap(1);

    auto multOp = rewriter.create<linalg::GenericOp>(
        loc, paramType, ValueRange{var, gamma}, empty1D,
        SmallVector<AffineMap>{map1D, map1D, map1D},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value a = b.create<arith::AddFOp>(l, args[0], eps);
          Value s = b.create<math::SqrtOp>(l, a);
          Value m = b.create<arith::DivFOp>(l, args[1], s);
          b.create<linalg::YieldOp>(l, m);
        });
    Value factor = multOp.getResult(0);

    // 2. Calculate New Bias
    auto biasOp = rewriter.create<linalg::GenericOp>(
        loc, paramType, ValueRange{beta, factor, mean}, empty1D,
        SmallVector<AffineMap>{map1D, map1D, map1D, map1D},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value prod = b.create<arith::MulFOp>(l, args[1], args[2]);
          Value res  = b.create<arith::SubFOp>(l, args[0], prod);
          b.create<linalg::YieldOp>(l, res);
        });
    Value newBias = biasOp.getResult(0);

    // 3. Fold Weights
    auto map4D = rewriter.getMultiDimIdentityMap(4);
    auto mapBroadcast = AffineMap::get(4, 0, {rewriter.getAffineDimExpr(0)}, rewriter.getContext());
    Value empty4D = rewriter.create<tensor::EmptyOp>(loc, weightType.getShape(), elementType);
    
    auto weightFoldOp = rewriter.create<linalg::GenericOp>(
        loc, weightType, ValueRange{weights, factor}, empty4D,
        SmallVector<AffineMap>{map4D, mapBroadcast, map4D},
        SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value res = b.create<arith::MulFOp>(l, args[0], args[1]);
          b.create<linalg::YieldOp>(l, res);
        });

    // 4. Update the Convolution Weights In-Place
    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp->setOperand(1, weightFoldOp.getResult(0));
    });

    // 5. Apply Bias AFTER Conv
    // Move the "pen" to just after the Convolution so we can add the bias to its result.
    rewriter.setInsertionPointAfter(convOp);

    auto mapData = rewriter.getMultiDimIdentityMap(4);
    auto mapBias = AffineMap::get(4, 0, {rewriter.getAffineDimExpr(3)}, rewriter.getContext());
    Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, outType.getShape(), elementType);

    auto finalAddOp = rewriter.create<linalg::GenericOp>(
        loc, outType, ValueRange{convOp->getResult(0), newBias}, emptyOut,
        SmallVector<AffineMap>{mapData, mapBias, mapData},
        SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value res = b.create<arith::AddFOp>(l, args[0], args[1]);
          b.create<linalg::YieldOp>(l, res);
        });

    // D. Final Replace
    rewriter.replaceOp(op, finalAddOp->getResults());
    return success();
  }
};

struct FoldBatchNormPass 
    : public PassWrapper<FoldBatchNormPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldBatchNormPass)
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldBatchNormPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  llvm::StringRef getArgument() const final { return "fold-batchnorm"; }
};

} // namespace

namespace mlir {
void registerFoldBatchNormPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<FoldBatchNormPass>();
    });
}
}