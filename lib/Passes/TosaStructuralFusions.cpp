#include "TosaPatterns.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

// Helper to permute a 4D weight tensor [OC, KH, KW, IC]
DenseElementsAttr permuteWeights(DenseElementsAttr attr, ArrayRef<int32_t> perms) {
    auto type = attr.getType().cast<RankedTensorType>();
    auto shape = type.getShape();
    SmallVector<float> oldValues(attr.getValues<float>());
    
    // Weights are [OC, KH, KW, IC]. 
    // If input is [N, H, W, C] and perms is [0, 2, 1, 3] (H/W swap),
    // we must swap KH (dim 1) and KW (dim 2).
    SmallVector<int64_t, 4> newShape = {
        shape[0],         // OC
        shape[perms[1]],  // New KH
        shape[perms[2]],  // New KW
        shape[perms[3]]   // New IC
    };

    SmallVector<float> newValues(oldValues.size());
    
    for (int64_t oc = 0; oc < shape[0]; ++oc) {
        for (int64_t kh = 0; kh < shape[1]; ++kh) {
            for (int64_t kw = 0; kw < shape[2]; ++kw) {
                for (int64_t ic = 0; ic < shape[3]; ++ic) {
                    // Original index in [OC, KH, KW, IC]
                    int64_t oldIdx = ((oc * shape[1] + kh) * shape[2] + kw) * shape[3] + ic;
                    
                    // Permute coordinates based on the transpose perms
                    int64_t coords[4] = {oc, kh, kw, ic};
                    int64_t newCoords[4];
                    newCoords[0] = oc;
                    newCoords[1] = coords[perms[1]];
                    newCoords[2] = coords[perms[2]];
                    newCoords[3] = coords[perms[3]];

                    int64_t newIdx = ((newCoords[0] * newShape[1] + newCoords[1]) * newShape[2] + newCoords[2]) * newShape[3] + newCoords[3];
                    newValues[newIdx] = oldValues[oldIdx];
                }
            }
        }
    }

    auto newType = RankedTensorType::get(newShape, type.getElementType());
    return DenseElementsAttr::get(newType, llvm::ArrayRef<float>(newValues));
}

struct TransposeFolding : public OpRewritePattern<tosa::Conv2DOp> {
    using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, 
                                 PatternRewriter &rewriter) const override {
        auto transposeOp = convOp.getInput().getDefiningOp<tosa::TransposeOp>();
        if (!transposeOp) return failure();

        auto permsConst = transposeOp.getPerms().getDefiningOp<tosa::ConstOp>();
        if (!permsConst) return failure();
        
        auto permsAttr = permsConst.getValue().cast<DenseElementsAttr>();
        auto perms = llvm::to_vector<4>(permsAttr.getValues<int32_t>());

        auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
        if (!weightConst) return failure();
        
        auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
        auto newWeightAttr = permuteWeights(weightAttr, perms);
        auto newWeightConst = rewriter.create<tosa::ConstOp>(
            weightConst.getLoc(), newWeightAttr.getType(), newWeightAttr);

        rewriter.replaceOpWithNewOp<tosa::Conv2DOp>(
            convOp, 
            convOp.getType(), 
            transposeOp.getInput1(), 
            newWeightConst,
            convOp.getBias(),
            convOp.getPadAttr(),
            convOp.getStrideAttr(),
            convOp.getDilationAttr()
        );

        return success();
    }
};

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

void mlir::tensormorph::populateTosaStructuralFusionPatterns(RewritePatternSet &patterns, bool fuseFanout, bool fuseActivations, bool fuseTranspose) {
  patterns.add<FoldLinearMathIntoConv>(patterns.getContext(), fuseFanout);
  if (fuseTranspose) {
    patterns.add<TransposeFolding>(patterns.getContext());
  }
  if (fuseActivations) {
    patterns.add<FuseClampIntoConv>(patterns.getContext());
  }
}