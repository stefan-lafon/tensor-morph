#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

namespace {

struct IterativeFoldBatchNorm : public OpRewritePattern<tosa::Conv2DOp> {
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, 
                                PatternRewriter &rewriter) const override {
    
    // 1. We MUST have constant weights AND a constant bias to fold these adjustments.
    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    
    if (!weightConst || !biasConst) return failure();

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();

    // Copy original values into mutable buffers
    SmallVector<float> currentWeights(weightAttr.getValues<float>());
    SmallVector<float> currentBias(biasAttr.getValues<float>());
    
    int numOutChannels = biasAttr.getNumElements();
    int weightsPerChannel = currentWeights.size() / numOutChannels;

    Operation *lastOp = convOp;
    bool changed = false;

    // 2. Greedy Forward Walk: Look at the users of the current result
    while (lastOp->getResult(0).hasOneUse()) {
      Operation *nextOp = *lastOp->getResult(0).getUsers().begin();

      // Handle Scaling (Mul)
      if (auto mulOp = llvm::dyn_cast<tosa::MulOp>(nextOp)) {
        auto scaleConst = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!scaleConst) break;
        
        auto sValues = scaleConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        
        for (int oc = 0; oc < numOutChannels; ++oc) {
          float s = sValues[oc];
          currentBias[oc] *= s;
          for (int i = 0; i < weightsPerChannel; ++i) {
            currentWeights[oc * weightsPerChannel + i] *= s;
          }
        }
        lastOp = nextOp;
        changed = true;
      } 
      // Handle Shifting (Add)
      else if (auto addOp = llvm::dyn_cast<tosa::AddOp>(nextOp)) {
        auto shiftConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!shiftConst) break;

        auto shValues = shiftConst.getValue().cast<DenseElementsAttr>().getValues<float>();

        for (int oc = 0; oc < numOutChannels; ++oc) {
          currentBias[oc] += shValues[oc];
        }
        lastOp = nextOp;
        changed = true;
      } 
      else {
        break; 
      }
    }

    if (!changed) return failure();

    // 3. Update the IR
    auto newWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(currentWeights));
    auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(currentBias));
    
    auto newWConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), weightAttr.getType(), newWAttr);
    auto newBConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), biasAttr.getType(), newBAttr);

    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.setOperand(1, newWConst);
      convOp.setOperand(2, newBConst);
    });

    // Replace the end of the chain with the output of the newly adjusted Conv
    rewriter.replaceOp(lastOp, convOp.getResult());
    
    return success();
  }
};

struct TosaFoldBatchNormPass : 
    public PassWrapper<TosaFoldBatchNormPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaFoldBatchNormPass)
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<IterativeFoldBatchNorm>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  
  llvm::StringRef getArgument() const final { return "fold-batchnorm"; }
};

} // namespace

namespace mlir {
void registerFoldBatchNormPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<TosaFoldBatchNormPass>();
    });
}
}