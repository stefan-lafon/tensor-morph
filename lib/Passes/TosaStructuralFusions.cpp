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
    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    if (!weightConst || !biasConst) return failure();

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();
    
    int numOutChannels = biasAttr.getNumElements();
    int weightsPerChannel = weightAttr.getNumElements() / numOutChannels;

    for (auto &use : convOp->getResult(0).getUses()) {
      Operation *nextOp = use.getOwner();
      bool isSingleUse = convOp->getResult(0).hasOneUse();

      if (!isSingleUse && !allowFanout) continue;

      // Handle Add
      if (auto addOp = llvm::dyn_cast<tosa::AddOp>(nextOp)) {
        auto shiftConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!shiftConst) continue;

        auto shValues = shiftConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> newBias(biasAttr.getValues<float>());
        for (int i = 0; i < numOutChannels; ++i) newBias[i] += shValues[i];

        auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(newBias));
        auto newBConst = rewriter.create<tosa::ConstOp>(addOp.getLoc(), biasAttr.getType(), newBAttr);

        if (isSingleUse) {
          rewriter.modifyOpInPlace(convOp, [&]() { convOp.setOperand(2, newBConst); });
          rewriter.replaceOp(addOp, convOp.getResult());
        } else {
          Operation *clonedConv = rewriter.clone(*convOp.getOperation());
          clonedConv->setOperand(2, newBConst);
          rewriter.replaceOp(addOp, clonedConv->getResult(0));
        }
        return success();
      }

      // Handle Mul
      if (auto mulOp = llvm::dyn_cast<tosa::MulOp>(nextOp)) {
        auto scaleConst = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!scaleConst) continue;

        auto sValues = scaleConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> newWeights(weightAttr.getValues<float>());
        SmallVector<float> newBias(biasAttr.getValues<float>());

        for (int oc = 0; oc < numOutChannels; ++oc) {
          float s = sValues[oc];
          newBias[oc] *= s;
          for (int i = 0; i < weightsPerChannel; ++i)
            newWeights[oc * weightsPerChannel + i] *= s;
        }

        auto newWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(newWeights));
        auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(newBias));
        auto newWConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), weightAttr.getType(), newWAttr);
        auto newBConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), biasAttr.getType(), newBAttr);

        if (isSingleUse) {
          rewriter.modifyOpInPlace(convOp, [&]() {
            convOp.setOperand(1, newWConst);
            convOp.setOperand(2, newBConst);
          });
          rewriter.replaceOp(mulOp, convOp.getResult());
        } else {
          Operation *clonedConv = rewriter.clone(*convOp.getOperation());
          clonedConv->setOperand(1, newWConst);
          clonedConv->setOperand(2, newBConst);
          rewriter.replaceOp(mulOp, clonedConv->getResult(0));
        }
        return success();
      }

      // Handle Sub (Restored!)
      if (auto subOp = llvm::dyn_cast<tosa::SubOp>(nextOp)) {
        auto subConst = subOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!subConst) continue;

        auto sValues = subConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> newBias(biasAttr.getValues<float>());
        for (int i = 0; i < numOutChannels; ++i) newBias[i] -= sValues[i];

        auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(newBias));
        auto newBConst = rewriter.create<tosa::ConstOp>(subOp.getLoc(), biasAttr.getType(), newBAttr);

        if (isSingleUse) {
          rewriter.modifyOpInPlace(convOp, [&]() { convOp.setOperand(2, newBConst); });
          rewriter.replaceOp(subOp, convOp.getResult());
        } else {
          Operation *clonedConv = rewriter.clone(*convOp.getOperation());
          clonedConv->setOperand(2, newBConst);
          rewriter.replaceOp(subOp, clonedConv->getResult(0));
        }
        return success();
      }
    }

    return failure();
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