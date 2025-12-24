/*
 * This file implements a set of greedy optimization patterns for the TOSA dialect.
 * The logic here focuses on reducing graph depth by merging pointwise operations
 * into preceding convolutions.
 * * Supported Merges:
 * - Linear Math Folding: tosa.add, tosa.sub, tosa.mul (folded into weights/bias)
 * - Activation Fusion: tosa.clamp (fused via "fused_activation" attribute)
 *
 * This reduces the instruction count and allows the backend to generate more 
 * efficient, single-kernel execution loops.
 */

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

namespace {

/**
 * Pattern 1: Linear Folding
 * Anchors on a Conv2D and consumes subsequent Add, Sub, and Mul operations.
 */
struct FoldLinearMathIntoConv : public OpRewritePattern<tosa::Conv2DOp> {
  using OpRewritePattern<tosa::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp convOp, 
                                PatternRewriter &rewriter) const override {
    
    // We need defined constants for weights and bias to perform folding.
    auto weightConst = convOp.getWeight().getDefiningOp<tosa::ConstOp>();
    auto biasConst = convOp.getBias().getDefiningOp<tosa::ConstOp>();
    
    if (!weightConst || !biasConst) return failure();

    auto weightAttr = weightConst.getValue().cast<DenseElementsAttr>();
    auto biasAttr = biasConst.getValue().cast<DenseElementsAttr>();

    // Pull values into local vectors so we can mutate them during the walk.
    SmallVector<float> currentWeights(weightAttr.getValues<float>());
    SmallVector<float> currentBias(biasAttr.getValues<float>());
    
    int numOutChannels = biasAttr.getNumElements();
    int weightsPerChannel = currentWeights.size() / numOutChannels;

    Operation *lastOp = convOp;
    bool changed = false;

    // Greedy forward walk through the IR. We only eat ops that have a single use.
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
      // Handle Shifting (Sub)
      else if (auto subOp = llvm::dyn_cast<tosa::SubOp>(nextOp)) {
        auto subConst = subOp.getInput2().getDefiningOp<tosa::ConstOp>();
        if (!subConst) break;

        auto sValues = subConst.getValue().cast<DenseElementsAttr>().getValues<float>();
        for (int oc = 0; oc < numOutChannels; ++oc) {
          currentBias[oc] -= sValues[oc];
        }
        lastOp = nextOp;
        changed = true;
      }
      else {
        break; 
      }
    }

    if (!changed) return failure();

    // Reify the mutated weights and bias back into the IR.
    auto newWAttr = DenseElementsAttr::get(weightAttr.getType(), llvm::ArrayRef<float>(currentWeights));
    auto newBAttr = DenseElementsAttr::get(biasAttr.getType(), llvm::ArrayRef<float>(currentBias));
    
    auto newWConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), weightAttr.getType(), newWAttr);
    auto newBConst = rewriter.create<tosa::ConstOp>(convOp.getLoc(), biasAttr.getType(), newBAttr);

    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.setOperand(1, newWConst);
      convOp.setOperand(2, newBConst);
    });

    // Reroute the final consumer to the updated convolution.
    rewriter.replaceOp(lastOp, convOp.getResult());
    return success();
  }
};

/**
 * Pattern 2: Clamp/Activation Fusion
 * Looks for Conv2D ops followed by a Clamp and tags the Conv with metadata.
 */
struct FuseClampIntoConv : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp clampOp, 
                                PatternRewriter &rewriter) const override {
    
    // Check if the input to this clamp is a Convolution
    auto convOp = clampOp.getInput().getDefiningOp<tosa::Conv2DOp>();
    if (!convOp) return failure();

    // Safety: don't fuse if the convolution result is branched to other logic.
    if (!convOp->getResult(0).hasOneUse()) return failure();

    // Check if we already fused something here.
    if (convOp->getAttr("fused_activation")) return failure();

    // Attach activation metadata as custom attributes for the backend kernel generator.
    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp->setAttr("fused_activation", rewriter.getStringAttr("clamp"));
      convOp->setAttr("clamp_min", clampOp.getMinIntAttr());
      rewriter.setInsertionPointAfter(convOp); // Keep IR clean
      convOp->setAttr("clamp_max", clampOp.getMaxIntAttr());
    });

    // Bypass the clamp operation.
    rewriter.replaceOp(clampOp, convOp.getResult());
    
    return success();
  }
};

/**
 * Pass Entry Point
 */
struct TosaOptimizationsPass : 
    public PassWrapper<TosaOptimizationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaOptimizationsPass)
  
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    
    patterns.add<FoldLinearMathIntoConv>(ctx);
    patterns.add<FuseClampIntoConv>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  
  llvm::StringRef getArgument() const final { return "tosa-opt"; }
};

} // namespace

namespace mlir {
void registerTosaOptimizationsPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return std::make_unique<TosaOptimizationsPass>();
    });
}
}