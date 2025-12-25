#include "TosaPatterns.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

/**
 * Helper: Permutes a 4D weight tensor [OC, KH, KW, IC] based on input perms.
 */
DenseElementsAttr permuteWeights(DenseElementsAttr attr, ArrayRef<int32_t> perms) {
    auto type = attr.getType().cast<RankedTensorType>();
    auto shape = type.getShape();
    SmallVector<float> oldValues(attr.getValues<float>());
    
    SmallVector<int64_t, 4> newShape = {
        shape[0],         
        shape[perms[1]],  
        shape[perms[2]],  
        shape[perms[3]]   
    };

    SmallVector<float> newValues(oldValues.size());
    for (int64_t oc = 0; oc < shape[0]; ++oc) {
        for (int64_t kh = 0; kh < shape[1]; ++kh) {
            for (int64_t kw = 0; kw < shape[2]; ++kw) {
                for (int64_t ic = 0; ic < shape[3]; ++ic) {
                    int64_t oldIdx = ((oc * shape[1] + kh) * shape[2] + kw) * shape[3] + ic;
                    int64_t coords[4] = {oc, kh, kw, ic};
                    int64_t newCoords[4] = {oc, coords[perms[1]], coords[perms[2]], coords[perms[3]]};

                    int64_t newIdx = ((newCoords[0] * newShape[1] + newCoords[1]) * newShape[2] + newCoords[2]) * newShape[3] + newCoords[3];
                    newValues[newIdx] = oldValues[oldIdx];
                }
            }
        }
    }

    auto newType = RankedTensorType::get(newShape, type.getElementType());
    return DenseElementsAttr::get(newType, llvm::ArrayRef<float>(newValues));
}

// 1. Pad Elimination
struct PadElimination : public OpRewritePattern<tosa::Conv2DOp> {
    using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {
        auto padOp = convOp.getInput().getDefiningOp<tosa::PadOp>();
        if (!padOp) return failure();

        auto padConst = padOp.getPadding().getDefiningOp<tosa::ConstOp>();
        if (!padConst) return failure();

        auto padValues = padConst.getValue().cast<DenseElementsAttr>().getValues<int32_t>();
        int64_t pHTop = padValues[2], pHBot = padValues[3], pWTop = padValues[4], pWBot = padValues[5];

        auto existingPad = convOp.getPad(); 
        SmallVector<int64_t, 4> newPadValues = {
            existingPad[0] + pHTop, existingPad[1] + pHBot,
            existingPad[2] + pWTop, existingPad[3] + pWBot
        };

        auto newConv = rewriter.create<tosa::Conv2DOp>(
            convOp.getLoc(), convOp.getType(), padOp.getInput1(), convOp.getWeight(), 
            convOp.getBias(), rewriter.getDenseI64ArrayAttr(newPadValues),
            convOp.getStrideAttr(), convOp.getDilationAttr()
        );
        
        // CRITICAL: Preserve fused activation attributes
        newConv->setAttrs(convOp->getAttrs());
        // But overwrite the pad attribute with the new merged values
        newConv.setPadAttr(rewriter.getDenseI64ArrayAttr(newPadValues));

        rewriter.replaceOp(convOp, newConv.getResult());
        return success();
    }
};

// 2. Transpose Folding
struct TransposeFolding : public OpRewritePattern<tosa::Conv2DOp> {
    using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {
        auto transposeOp = convOp.getInput().getDefiningOp<tosa::TransposeOp>();
        if (!transposeOp) return failure();

        auto permsConst = transposeOp.getPerms().getDefiningOp<tosa::ConstOp>();
        if (!permsConst) return failure();
        
        auto perms = llvm::to_vector<4>(permsConst.getValue().cast<DenseElementsAttr>().getValues<int32_t>());
        auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
        if (!weightConst) return failure();
        
        auto newWeightAttr = permuteWeights(weightConst.getValue().cast<DenseElementsAttr>(), perms);
        auto newWeightConst = rewriter.create<tosa::ConstOp>(weightConst.getLoc(), newWeightAttr.getType(), newWeightAttr);

        auto newConv = rewriter.create<tosa::Conv2DOp>(
            convOp.getLoc(), convOp.getType(), transposeOp.getInput1(), newWeightConst,
            convOp.getBias(), convOp.getPadAttr(), convOp.getStrideAttr(), convOp.getDilationAttr()
        );
        
        // CRITICAL: Preserve fused activation attributes
        newConv->setAttrs(convOp->getAttrs());

        rewriter.replaceOp(convOp, newConv.getResult());
        return success();
    }
};

// 3. Linear Math Folding (The Eating Machine)
struct FoldLinearMathIntoConv : public OpRewritePattern<tosa::Conv2DOp> {
  bool allowFanout;
  FoldLinearMathIntoConv(MLIRContext *context, bool fuseFanout) 
      : OpRewritePattern<tosa::Conv2DOp>(context), allowFanout(fuseFanout) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {
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

      // --- Handle Add ---
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
          // CLONE ensures all attributes (activation fusion) are kept
          auto cloned = cast<tosa::Conv2DOp>(rewriter.clone(*convOp.getOperation()));
          cloned.setOperand(2, newBConst);
          rewriter.replaceOp(addOp, cloned.getResult());
        }
        return success();
      }

      // --- Handle Mul ---
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
          auto cloned = cast<tosa::Conv2DOp>(rewriter.clone(*convOp.getOperation()));
          cloned.setOperand(1, newWConst);
          cloned.setOperand(2, newBConst);
          rewriter.replaceOp(mulOp, cloned.getResult());
        }
        return success();
      }

      // --- Handle Sub ---
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
          auto cloned = cast<tosa::Conv2DOp>(rewriter.clone(*convOp.getOperation()));
          cloned.setOperand(2, newBConst);
          rewriter.replaceOp(subOp, cloned.getResult());
        }
        return success();
      }
    }
    return failure();
  }
};

// 4. Activation Injection
struct FuseClampIntoConv : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp clampOp, PatternRewriter &rewriter) const override {
    auto convOp = clampOp.getInput().getDefiningOp<tosa::Conv2DOp>();
    if (!convOp || !convOp->getResult(0).hasOneUse() || convOp->getAttr("fused_activation")) return failure();

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

void mlir::tensormorph::populateTosaStructuralFusionPatterns(
    RewritePatternSet &patterns, bool fuseFanout, bool fuseActivations, 
    bool fuseTranspose, bool fusePadding, bool fuseLinear) {
  
  if (fuseLinear) {
    patterns.add<FoldLinearMathIntoConv>(patterns.getContext(), fuseFanout);
  }
  if (fusePadding) patterns.add<PadElimination>(patterns.getContext());
  if (fuseTranspose) patterns.add<TransposeFolding>(patterns.getContext());
  if (fuseActivations) patterns.add<FuseClampIntoConv>(patterns.getContext());
}