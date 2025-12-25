#include "TosaPatterns.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

struct FoldLinearMathIntoConv : public OpRewritePattern<tosa::Conv2DOp> {
  bool allowFanout;
  FoldLinearMathIntoConv(MLIRContext *context, bool fuseFanout) 
      : OpRewritePattern<tosa::Conv2DOp>(context), allowFanout(fuseFanout) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, 
                                PatternRewriter &rewriter) const override {
    // Current "Polite" logic: respects the fanout flag
    if (!allowFanout && !convOp->getResult(0).hasOneUse()) 
        return failure();

    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    if (!weightConst || !biasConst) return failure();

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();

    SmallVector<float> currentWeights(weightAttr.getValues<float>());
    SmallVector<float> currentBias(biasAttr.getValues<float>());
    
    int numOutChannels = biasAttr.getNumElements();
    int weightsPerChannel = currentWeights.size() / numOutChannels;

    Operation *lastOp = convOp;
    bool changed = false;

    while (lastOp->getResult(0).hasOneUse()) {
      Operation *nextOp = *lastOp->getResult(0).getUsers().begin();

      if (auto mulOp = llvm::dyn_cast<tosa::MulOp>(nextOp)) {
        auto scaleConst = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!scaleConst) break;
        auto sValues = scaleConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) {
          float s = sValues[oc];
          currentBias[oc] *= s;
          for (int i = 0; i < weightsPerChannel; ++i) 
              currentWeights[oc * weightsPerChannel + i] *= s;
        }
        lastOp = nextOp; changed = true;
      } 
      else if (auto addOp = llvm::dyn_cast<tosa::AddOp>(nextOp)) {
        auto shiftConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!shiftConst) break;
        auto shValues = shiftConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) currentBias[oc] += shValues[oc];
        lastOp = nextOp; changed = true;
      } 
      else if (auto subOp = llvm::dyn_cast<tosa::SubOp>(nextOp)) {
        auto subConst = subOp.getOperand(1).getDefiningOp<tosa::ConstOp>();
        if (!subConst) break;
        auto sValues = subConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) currentBias[oc] -= sValues[oc];
        lastOp = nextOp; changed = true;
      }
      else break; 
    }

    if (!changed) return failure();

    auto newWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(currentWeights));
    auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(currentBias));
    auto newWConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), weightAttr.getType(), newWAttr);
    auto newBConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), biasAttr.getType(), newBAttr);

    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.setOperand(1, newWConst);
      convOp.setOperand(2, newBConst);
    });

    rewriter.replaceOp(lastOp, convOp.getResult());
    return success();
  }
};

struct FuseClampIntoConv : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp clampOp, 
                                PatternRewriter &rewriter) const override {
    auto convOp = clampOp.getInput().getDefiningOp<tosa::Conv2DOp>();
    if (!convOp || !convOp->getResult(0).hasOneUse() || convOp->getAttr("fused_activation")) 
        return failure();

    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp->setAttr("fused_activation", rewriter.getStringAttr("clamp"));
      convOp->setAttr("clamp_min", clampOp.getMinIntAttr());
      convOp->setAttr("clamp_max", clampOp.getMaxIntAttr());
    });

    rewriter.replaceOp(clampOp, convOp.getResult());
    return success();
  }
};

} // namespace

void mlir::tensormorph::populateTosaStructuralFusionPatterns(RewritePatternSet &patterns, bool fuseFanout, bool fuseActivations) {
  patterns.add<FoldLinearMathIntoConv>(patterns.getContext(), fuseFanout);
  if (fuseActivations) {
    patterns.add<FuseClampIntoConv>(patterns.getContext());
  }
}