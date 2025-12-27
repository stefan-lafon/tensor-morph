#include "TosaPatterns.h"
#include "TensorFeatures.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tensormorph;

namespace {

/**
 * Helper: Permutes a 4D weight tensor [OC, KH, KW, IC] based on input perms.
 * Used to fold transposes into convolution weights at compile-time.
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

// 1. Pad Elimination.
// Absorbs explicit tosa.pad operations into the internal convolution padding attribute.
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
        
        newConv->setAttrs(convOp->getAttrs());
        newConv.setPadAttr(rewriter.getDenseI64ArrayAttr(newPadValues));

        rewriter.replaceOp(convOp, newConv.getResult());
        return success();
    }
};

// 2. Transpose Folding.
// Bakes spatial permutations directly into the weight constant.
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
        
        newConv->setAttrs(convOp->getAttrs());

        rewriter.replaceOp(convOp, newConv.getResult());
        return success();
    }
};

// 3. Linear Math Folding (Standard Conv2D).
// Fuses multiplication, addition, and subtraction into weights/biases.
struct FoldLinearMathIntoConv : public OpRewritePattern<tosa::Conv2DOp> {
  bool allowFanout;
  Advisor *advisor;
  float minProfit;
  bool debugAi;

  FoldLinearMathIntoConv(MLIRContext *context, bool fuseFanout, Advisor *adv, float profit, bool debug) 
      : OpRewritePattern<tosa::Conv2DOp>(context), allowFanout(fuseFanout), advisor(adv), minProfit(profit), debugAi(debug) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, PatternRewriter &rewriter) const override {
    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    if (!weightConst || !biasConst) return failure();

    // AI Decision Point: Consult the Advisor if active.
    // Fusion decision logic:
    // 1. If no advisor is provided, we default to a greedy policy (always fuse).
    // 2. If an advisor is active, we extract IR features and query the model.
    // 3. Fusions are only applied if the predicted profit meets the user-defined threshold.
    if (advisor) {
      TensorFeatures features = extractFeatures(convOp);
      float predictedProfit = advisor->Predict(features.toVector());
      
      if (debugAi) {
        llvm::errs() << "[AI Decision] Profile: " << advisor->GetProfileName() 
                     << " | Score: " << predictedProfit << "\n";
      }

      if (predictedProfit < minProfit) {
        return failure(); 
      }
    }

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();
    int numOutChannels = biasAttr.getNumElements();

    for (auto &use : llvm::make_early_inc_range(convOp->getResult(0).getUses())) {
      Operation *nextOp = use.getOwner();
      bool isSingleUse = convOp->getResult(0).hasOneUse();
      if (!isSingleUse && !allowFanout) continue;

      if (auto addOp = llvm::dyn_cast<tosa::AddOp>(nextOp)) {
        auto constOp = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!constOp) continue;
        auto vals = constOp.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> nB(biasAttr.getValues<float>());
        for (int i = 0; i < numOutChannels; ++i) nB[i] += vals[i];
        auto nBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(nB));
        auto nBConst = rewriter.create<tosa::ConstOp>(addOp.getLoc(), biasAttr.getType(), nBAttr);
        
        auto cloned = cast<tosa::Conv2DOp>(rewriter.clone(*convOp.getOperation()));
        cloned.setOperand(2, nBConst);
        rewriter.replaceOp(addOp, cloned.getResult());
        if (isSingleUse) rewriter.eraseOp(convOp);
        return success();
      }

      if (auto subOp = llvm::dyn_cast<tosa::SubOp>(nextOp)) {
        auto constOp = subOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!constOp) continue;
        auto vals = constOp.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> nB(biasAttr.getValues<float>());
        for (int i = 0; i < numOutChannels; ++i) nB[i] -= vals[i];
        auto nBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(nB));
        auto nBConst = rewriter.create<tosa::ConstOp>(subOp.getLoc(), biasAttr.getType(), nBAttr);
        
        auto cloned = cast<tosa::Conv2DOp>(rewriter.clone(*convOp.getOperation()));
        cloned.setOperand(2, nBConst);
        rewriter.replaceOp(subOp, cloned.getResult());
        if (isSingleUse) rewriter.eraseOp(convOp);
        return success();
      }

      if (auto mulOp = llvm::dyn_cast<tosa::MulOp>(nextOp)) {
        auto constOp = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!constOp) continue;
        auto vals = constOp.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> nW(weightAttr.getValues<float>());
        SmallVector<float> nB(biasAttr.getValues<float>());
        int wpc = weightAttr.getNumElements() / numOutChannels;
        for (int oc = 0; oc < numOutChannels; ++oc) {
          float s = vals[oc];
          nB[oc] *= s;
          for (int i = 0; i < wpc; ++i) nW[oc * wpc + i] *= s;
        }
        auto nWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(nW));
        auto nBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(nB));
        auto nWConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), weightAttr.getType(), nWAttr);
        auto nBConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), biasAttr.getType(), nBAttr);
        
        auto cloned = cast<tosa::Conv2DOp>(rewriter.clone(*convOp.getOperation()));
        cloned.setOperand(1, nWConst);
        cloned.setOperand(2, nBConst);
        rewriter.replaceOp(mulOp, cloned.getResult());
        if (isSingleUse) rewriter.eraseOp(convOp);
        return success();
      }
    }
    return failure();
  }
};

// 4. Linear Math Folding (Depthwise Conv2D).
struct FoldLinearMathIntoDepthwiseConv : public OpRewritePattern<tosa::DepthwiseConv2DOp> {
  bool allowFanout;
  Advisor *advisor;
  float minProfit;
  bool debugAi;

  FoldLinearMathIntoDepthwiseConv(MLIRContext *context, bool fuseFanout, Advisor *adv, float profit, bool debug) 
      : OpRewritePattern<tosa::DepthwiseConv2DOp>(context), allowFanout(fuseFanout), advisor(adv), minProfit(profit), debugAi(debug) {}

  LogicalResult matchAndRewrite(tosa::DepthwiseConv2DOp dwOp, PatternRewriter &rewriter) const override {
    auto weightConst = dwOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = dwOp.getBias().getDefiningOp<tosa::ConstOp>();
    if (!weightConst || !biasConst) return failure();

    if (advisor) {
      TensorFeatures features = extractFeatures(dwOp);
      float predictedProfit = advisor->Predict(features.toVector());
      
      if (debugAi) {
        llvm::errs() << "[AI Decision] Profile: " << advisor->GetProfileName() 
                     << " | Score: " << predictedProfit << "\n";
      }

      if (predictedProfit < minProfit) {
        return failure();
      }
    }

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();
    int numOutChannels = biasAttr.getNumElements();
    auto weightShape = weightAttr.getType().cast<RankedTensorType>().getShape();
    int kh = weightShape[0], kw = weightShape[1];

    for (auto &use : llvm::make_early_inc_range(dwOp->getResult(0).getUses())) {
      Operation *nextOp = use.getOwner();
      bool isSingleUse = dwOp->getResult(0).hasOneUse();
      if (!isSingleUse && !allowFanout) continue;

      if (auto addOp = llvm::dyn_cast<tosa::AddOp>(nextOp)) {
        auto constOp = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!constOp) continue;
        auto vals = constOp.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> nB(biasAttr.getValues<float>());
        for (int i = 0; i < numOutChannels; ++i) nB[i] += vals[i];
        auto nBConst = rewriter.create<tosa::ConstOp>(addOp.getLoc(), biasAttr.getType(), DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(nB)));
        auto cloned = cast<tosa::DepthwiseConv2DOp>(rewriter.clone(*dwOp.getOperation()));
        cloned.setOperand(2, nBConst);
        rewriter.replaceOp(addOp, cloned.getResult());
        if (isSingleUse) rewriter.eraseOp(dwOp);
        return success();
      }

      if (auto mulOp = llvm::dyn_cast<tosa::MulOp>(nextOp)) {
        auto constOp = mulOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!constOp) continue;
        auto vals = constOp.getValue().cast<DenseElementsAttr>().getValues<float>();
        SmallVector<float> nW(weightAttr.getValues<float>());
        SmallVector<float> nB(biasAttr.getValues<float>());
        for (int oc = 0; oc < numOutChannels; ++oc) {
          float s = vals[oc];
          nB[oc] *= s;
          for (int h = 0; h < kh; ++h) {
            for (int w = 0; w < kw; ++w) {
              int idx = ((h * kw + w) * numOutChannels) + oc;
              nW[idx] *= s;
            }
          }
        }
        auto nWConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), weightAttr.getType(), DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(nW)));
        auto nBConst = rewriter.create<tosa::ConstOp>(mulOp.getLoc(), biasAttr.getType(), DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(nB)));
        auto cloned = cast<tosa::DepthwiseConv2DOp>(rewriter.clone(*dwOp.getOperation()));
        cloned.setOperand(1, nWConst);
        cloned.setOperand(2, nBConst);
        rewriter.replaceOp(mulOp, cloned.getResult());
        if (isSingleUse) rewriter.eraseOp(dwOp);
        return success();
      }
    }
    return failure();
  }
};

// 5. Activation Injection.
// Fuses Clamp operations into convolution anchors as fused attributes.
struct FuseClampIntoAnchor : public OpRewritePattern<tosa::ClampOp> {
    using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tosa::ClampOp clampOp, PatternRewriter &rewriter) const override {
        Operation *anchor = clampOp.getInput().getDefiningOp();
        if (!anchor || !isa<tosa::Conv2DOp, tosa::DepthwiseConv2DOp>(anchor)) return failure();
        if (!anchor->getResult(0).hasOneUse() || anchor->hasAttr("fused_activation")) return failure();

        rewriter.modifyOpInPlace(anchor, [&]() {
          anchor->setAttr("fused_activation", rewriter.getStringAttr("clamp"));
          anchor->setAttr("clamp_min", clampOp.getMinIntAttr());
          anchor->setAttr("clamp_max", clampOp.getMaxIntAttr());
        });
        rewriter.replaceOp(clampOp, anchor->getResult(0));
        return success();
    }
};

} // namespace

void mlir::tensormorph::populateTosaStructuralFusionPatterns(
    RewritePatternSet &patterns, Advisor *advisor, float minProfit,
    bool fuseFanout, bool fuseActivations, 
    bool fuseTranspose, bool fusePadding, bool fuseLinear, bool debugAi) {
  
  if (fuseLinear) {
    patterns.add<FoldLinearMathIntoConv>(patterns.getContext(), fuseFanout, advisor, minProfit, debugAi);
    patterns.add<FoldLinearMathIntoDepthwiseConv>(patterns.getContext(), fuseFanout, advisor, minProfit, debugAi);
  }
  if (fusePadding) patterns.add<PadElimination>(patterns.getContext());
  if (fuseTranspose) patterns.add<TransposeFolding>(patterns.getContext());
  if (fuseActivations) patterns.add<FuseClampIntoAnchor>(patterns.getContext());
}