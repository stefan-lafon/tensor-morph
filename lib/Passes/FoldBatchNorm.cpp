// @title FoldBatchNorm.cpp
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

namespace {
// This pattern implements the classic 'BatchNorm Folding' into a 
// Convolution. We specifically target the TOSA dialect.
struct FoldTosaBatchNorm : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp mulOp, 
                                PatternRewriter &rewriter) const override {
    
    // 1. Trace the consumer: We expect Mul -> Add (Scale -> Shift)
    auto convOp = mulOp.getInput1().getDefiningOp<tosa::Conv2DOp>();
    if (!convOp) return failure();

    // Check if the Mul result is consumed by an Add
    if (!mulOp.getResult().hasOneUse()) return failure();
    auto addOp = llvm::dyn_cast<tosa::AddOp>(*mulOp.getResult().getUsers().begin());
    if (!addOp) return failure();

    // 2. Extract weights and constants
    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    auto scaleConst = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
    auto shiftConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();

    if (!weightConst || !biasConst || !scaleConst || !shiftConst) return failure();

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto scaleAttr = scaleConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();
    auto shiftAttr = shiftConst.getValue().cast<DenseElementsAttr>();

    // 3. Mathematical Folding logic
    SmallVector<float> newWeights;
    auto wValues = weightAttr.getValues<float>();
    auto sValues = scaleAttr.getValues<float>();
    
    // Conv weight layout: [OutChan, K, K, InChan]
    int elementsPerChannel = weightAttr.getNumElements() / sValues.size();
    for (int oc = 0; oc < sValues.size(); ++oc) {
      float scale = sValues[oc];
      for (int i = 0; i < elementsPerChannel; ++i) {
        newWeights.push_back(wValues[oc * elementsPerChannel + i] * scale);
      }
    }

    SmallVector<float> newBias;
    auto bValues = biasAttr.getValues<float>();
    auto shValues = shiftAttr.getValues<float>();
    for (int i = 0; i < bValues.size(); ++i) {
      newBias.push_back((bValues[i] * sValues[i]) + shValues[i]);
    }

    // 4. In-place IR Update
    rewriter.setInsertionPoint(convOp);
    auto newWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(newWeights));
    auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(newBias));
    
    auto newWConst = rewriter.create<tosa::ConstOp>(weightConst.getLoc(), weightAttr.getType(), newWAttr);
    auto newBConst = rewriter.create<tosa::ConstOp>(biasConst.getLoc(), biasAttr.getType(), newBAttr);

    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.setOperand(1, newWConst);
      convOp.setOperand(2, newBConst);
    });

    rewriter.replaceOp(addOp, convOp.getResult());
    return success();
  }
};

struct TosaFoldBatchNormPass : 
    public PassWrapper<TosaFoldBatchNormPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaFoldBatchNormPass)
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FoldTosaBatchNorm>(ctx);
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